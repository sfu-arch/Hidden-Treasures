#!/usr/bin/env python3
"""
spgemm_mtx.py

Real-dataset counterpart to spgemm.py: computes C = A @ A^T (CSR x CSR -> CSR)
on the SuiteSparse SpGEMM matrices fetched by:

    python download_datasets.py --task spgemm --all

Those datasets land under MS/<name>/. download_datasets.py parses each
original .mtx into structure-only COO indices (edges.npy + metadata.txt) and
deletes the .mtx after parsing, so by default this script reads that format
(unit-weight / pattern-only matrix). If a raw .mtx is still present in a
dataset's directory (e.g. a download that didn't finish, or one fetched
separately), it is parsed directly instead and real NNZ values are used when
the file carries them. Same dataset loader as spmm_mtx.py.

Modes (structured like spgemm.py / masked_spgemm_roofline.py):
  timing   CUDA-event timing per dataset + self-calibrated roofline -> CSV
  profile  run ONE dataset inside an NVTX "measure" range, write a meta
           sidecar (drive with Nsight Compute; see run_profile.sh)
  plot     parse ncu CSV exports + meta -> SVG roofline scatter, each dataset
           placed at MEASURED arithmetic intensity (FLOP / DRAM byte) vs
           achieved TFLOPS, annotated with %peak SM and tensor-pipe util

Values are always computed in FP32. download_datasets.py's SuiteSparse loader
does not preserve NNZ values (structure-only), so A is a 0/1 pattern matrix
unless a raw .mtx with real values is found. cuSPARSE SpGEMM supports
float/double/complex only (no integer path) and requires 32-bit indices,
which are enforced on load.

Backend: CuPy's cusparseSpGEMM binding (cupyx.scipy.sparse csr @ csr).
Deps:    pip install cupy-cuda12x scipy numpy matplotlib   (match cupy to CUDA)

Examples
--------
python spgemm_mtx.py timing --ms-root MS --all
python spgemm_mtx.py timing --ms-root MS --datasets poisson3Da circuit_2 --out spgemm_mtx.csv

python spgemm_mtx.py profile --op poisson3Da --ms-root MS \
    --meta-out prof_spgemm_mtx/poisson3Da.meta.json

python spgemm_mtx.py plot --prof prof_spgemm_mtx --out spgemm_mtx_roofline.svg
"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

WARMUP, REPEAT = 3, 10

M_TIME   = "gpu__time_duration.sum"
M_BYTES  = "dram__bytes.sum"
M_SM     = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
M_DRAM   = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
M_TENSOR = "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"
NCU_METRICS = ",".join([M_TIME, M_BYTES, M_SM, M_DRAM, M_TENSOR])


# --------------------------------------------------------------------------- #
# dataset loading — MS/<name>/{edges.npy,metadata.txt} with raw-.mtx fallback
# (identical convention to spmm_mtx.py, duplicated here to keep each profiler
# script self-contained)
# --------------------------------------------------------------------------- #
def _read_metadata(meta_path: Path) -> dict:
    meta = {}
    with open(meta_path) as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = line.strip().split("=", 1)
            meta[k] = v
    return meta


def _parse_mtx(mtx_path: Path):
    """Parse a Matrix Market file. Returns (rows, cols, data, num_rows, num_cols).
    Uses real NNZ values when the file carries a 3rd column; otherwise unit
    weights (pattern matrix). Symmetric matrices are mirrored off-diagonal."""
    rows, cols, vals = [], [], []
    num_rows = num_cols = 0
    is_symmetric = False
    data_started = False
    with open(mtx_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("%%MatrixMarket"):
                is_symmetric = "symmetric" in line.lower()
                continue
            if line.startswith("%"):
                continue
            parts = line.split()
            if not data_started:
                num_rows, num_cols = int(parts[0]), int(parts[1])
                data_started = True
                continue
            r, c = int(parts[0]) - 1, int(parts[1]) - 1
            v = float(parts[2]) if len(parts) >= 3 else 1.0
            rows.append(r); cols.append(c); vals.append(v)
            if is_symmetric and r != c:
                rows.append(c); cols.append(r); vals.append(v)
    return (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64),
            np.array(vals, dtype=np.float32), num_rows, num_cols)


def load_dataset(name: str, ms_root: Path):
    """Return (rows, cols, data, num_rows, num_cols, meta) for MS/<name>/."""
    ds_dir = ms_root / name
    if not ds_dir.is_dir():
        raise FileNotFoundError(
            f"'{ds_dir}' not found. Fetch it with:\n"
            f"  python download_datasets.py --task spgemm --datasets {name}"
        )

    edges_path = ds_dir / "edges.npy"
    meta_path = ds_dir / "metadata.txt"
    if edges_path.exists():
        edges = np.load(edges_path)
        meta = _read_metadata(meta_path) if meta_path.exists() else {}
        num_rows = int(meta.get("num_rows", edges[:, 0].max() + 1))
        num_cols = int(meta.get("num_cols", edges[:, 1].max() + 1))
        rows, cols = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
        data = np.ones(len(rows), dtype=np.float32)   # structure-only (download_datasets.py drops values)
        return rows, cols, data, num_rows, num_cols, meta

    mtx_candidates = [
        p for p in ds_dir.rglob("*.mtx")
        if not (p.stem.endswith("_b") or p.stem.endswith("_x"))
    ]
    if mtx_candidates:
        rows, cols, data, num_rows, num_cols = _parse_mtx(mtx_candidates[0])
        return rows, cols, data, num_rows, num_cols, {}

    raise FileNotFoundError(
        f"No edges.npy or .mtx found under '{ds_dir}'. Fetch it with:\n"
        f"  python download_datasets.py --task spgemm --datasets {name}"
    )


def discover_datasets(ms_root: Path):
    names = []
    if ms_root.is_dir():
        for d in sorted(ms_root.iterdir()):
            if d.is_dir() and ((d / "edges.npy").exists() or list(d.rglob("*.mtx"))):
                names.append(d.name)
    return names


# --------------------------------------------------------------------------- #
# operands
# --------------------------------------------------------------------------- #
def products_count(A_csr, At_csr):
    """Exact multiply-add pairs in A@A^T = sum over each nonzero (i,k) of A of
    nnz(row k of A^T). FLOP basis = 2 * products."""
    rowlen_At = np.diff(At_csr.indptr).astype(np.int64)
    return int(rowlen_At[A_csr.indices.astype(np.int64)].sum())


def build_op(op, ms_root, dev_gb=None):
    """Return (fn, meta). fn() runs the op once. Anchors: gemm, copy."""
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import scipy.sparse as sp

    if op == "gemm":                     # compute-ceiling anchor (bf16 tensor)
        n = 8192
        a = cp.random.rand(n, n, dtype=cp.float32).astype(cp.float32)
        b = cp.random.rand(n, n, dtype=cp.float32).astype(cp.float32)
        fn = lambda: a @ b
        return fn, dict(op="gemm", flops=2 * n ** 3, bytes_model=2 * 3 * n * n,
                        nnz_A=n * n, dtype="fp32")

    if op == "copy":                     # bandwidth-ceiling anchor
        nbytes = 1 << 30
        src = cp.empty(nbytes // 4, dtype=cp.float32)
        dst = cp.empty_like(src)
        fn = lambda: cp.copyto(dst, src)
        return fn, dict(op="copy", flops=0, bytes_model=2 * nbytes, dtype="fp32")

    # dataset op: A @ A^T in fp32
    rows, cols, data, num_rows, num_cols, meta = load_dataset(op, ms_root)
    A_cpu = sp.coo_matrix((data, (rows, cols)), shape=(num_rows, num_cols)).tocsr()
    A_cpu.indices = A_cpu.indices.astype(np.int32)
    A_cpu.indptr = A_cpu.indptr.astype(np.int32)
    A_cpu.data = A_cpu.data.astype(np.float32)

    At_cpu = A_cpu.T.tocsr()
    At_cpu.indices = At_cpu.indices.astype(np.int32)
    At_cpu.indptr = At_cpu.indptr.astype(np.int32)
    prod = products_count(A_cpu, At_cpu)

    A = csp.csr_matrix((cp.asarray(A_cpu.data), cp.asarray(A_cpu.indices),
                        cp.asarray(A_cpu.indptr)), shape=A_cpu.shape)
    At = csp.csr_matrix((cp.asarray(At_cpu.data), cp.asarray(At_cpu.indices),
                         cp.asarray(At_cpu.indptr)), shape=At_cpu.shape)
    fn = lambda: A @ At
    out_meta = dict(op=op, flops=2 * prod, products=prod, nnz_A=int(A_cpu.nnz),
                    rows=int(A_cpu.shape[0]), cols=int(A_cpu.shape[1]),
                    dtype="fp32", degree_type=meta.get("degree_type", ""),
                    domain=meta.get("domain", ""))
    return fn, out_meta


# --------------------------------------------------------------------------- #
# TIMING
# --------------------------------------------------------------------------- #
def _time(fn):
    import cupy as cp
    s, e = cp.cuda.Event(), cp.cuda.Event()
    for _ in range(WARMUP):
        fn()
    cp.cuda.Device().synchronize()
    ts = []
    for _ in range(REPEAT):
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(cp.cuda.get_elapsed_time(s, e))   # ms
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def _check_result(nnz_c, meta):
    """Sanity-check a computed A@A^T result against the CPU-side exact
    product count. A nonzero `products` count with an empty (nnz==0) output
    means cuSPARSE returned silently -- no exception -- but the result is
    wrong. This happens when the exact number of intermediate multiply-add
    pairs exceeds cuSPARSE's internal 32-bit counters (~2.1e9), which very
    dense products (e.g. gupta3, ~30.7e9 products) can hit."""
    if meta.get("products", 0) > 0 and nnz_c == 0:
        return "SUSPECT_EMPTY (cuSPARSE silently returned nothing; likely internal overflow)"
    return "ok"


def calibrate(ms_root):
    fg, mg = build_op("gemm", ms_root)
    ms, _ = _time(fg)
    peak_tflops = mg["flops"] / (ms * 1e-3) / 1e12
    fc, mc = build_op("copy", ms_root)
    ms, _ = _time(fc)
    bw_gbs = mc["bytes_model"] / (ms * 1e-3) / 1e9
    return peak_tflops, bw_gbs, (peak_tflops * 1e12) / (bw_gbs * 1e9)


def run_timing(args):
    import cupy as cp
    ms_root = Path(args.ms_root)
    datasets = args.datasets or discover_datasets(ms_root)
    if not datasets:
        raise SystemExit(f"no datasets found under '{ms_root}' and none passed via --datasets")

    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"device: {name}  cupy {cp.__version__}")
    peak_tflops, bw_gbs, ridge = calibrate(ms_root)
    print(f"\n--- empirical roofline ---")
    print(f"compute ceiling : {peak_tflops:8.1f} TFLOPS (bf16 GEMM)")
    print(f"bandwidth       : {bw_gbs:8.1f} GB/s")
    print(f"ridge point     : {ridge:8.1f} flop/byte\n")

    fields = ["dataset", "rows", "nnz_A", "nnz_AAt", "products", "time_ms_median",
              "time_ms_min", "gflops", "pct_peak", "status", "degree_type", "domain"]
    out_rows = []
    print(f"{'dataset':>14}{'ms':>10}{'GFLOP/s':>10}{'%peak':>8}   nnz_A -> nnz(AA^T)")
    print("-" * 76)
    for ds in datasets:
        try:
            fn, meta = build_op(ds, ms_root)
        except FileNotFoundError as exc:
            print(f"{ds:>14}  skip ({exc})")
            continue
        try:
            med, mn = _time(fn)
            C = fn(); cp.cuda.Device().synchronize(); nnz_c = int(C.nnz); del C
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as exc:
            print(f"{ds:>14}  FAILED (cuSPARSE raised: {exc})")
            out_rows.append(dict(dataset=ds, rows=meta.get("rows"), nnz_A=meta.get("nnz_A"),
                                 nnz_AAt="", products=meta.get("products"), time_ms_median="",
                                 time_ms_min="", gflops="", pct_peak="", status=f"ERROR: {exc}",
                                 degree_type=meta.get("degree_type", ""), domain=meta.get("domain", "")))
            continue
        status = _check_result(nnz_c, meta)
        if status == "ok":
            gfl = (meta["flops"] / 1e9) / (med / 1e3)
            pct = 100 * (gfl / 1e3) / peak_tflops
            print(f"{ds:>14}{med:>10.3f}{gfl:>10.2f}{pct:>7.1f}%   {meta['nnz_A']:,} -> {nnz_c:,}")
        else:
            print(f"{ds:>14}{med:>10.3f}{'--':>10}{'--':>8}   {meta['nnz_A']:,} -> {nnz_c:,}   ** {status} **")
        out_rows.append(dict(dataset=ds, rows=meta["rows"], nnz_A=meta["nnz_A"], nnz_AAt=nnz_c,
                             products=meta["products"], time_ms_median=f"{med:.4f}",
                             time_ms_min=f"{mn:.4f}",
                             gflops=f"{gfl:.2f}" if status == "ok" else "",
                             pct_peak=f"{pct:.2f}" if status == "ok" else "",
                             status=status,
                             degree_type=meta["degree_type"], domain=meta["domain"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {args.out}")


# --------------------------------------------------------------------------- #
# PROFILE  (run under ncu)
# --------------------------------------------------------------------------- #
def run_profile(args):
    import cupy as cp
    ms_root = Path(args.ms_root)
    fn, meta = build_op(args.op, ms_root)
    for _ in range(WARMUP):
        fn()
    cp.cuda.Device().synchronize()
    cp.cuda.nvtx.RangePush("measure")        # ncu: --nvtx-include "measure/"
    try:
        out = fn()
        cp.cuda.Device().synchronize()
    except Exception as exc:
        cp.cuda.nvtx.RangePop()
        meta["status"] = f"ERROR: {exc}"
        if args.meta_out:
            os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)
            with open(args.meta_out, "w") as f:
                json.dump(meta, f, indent=2)
        raise
    cp.cuda.nvtx.RangePop()
    if hasattr(out, "nnz"):          # dataset ops return a sparse result; anchors don't
        meta["nnz_result"] = int(out.nnz)
        meta["status"] = _check_result(meta["nnz_result"], meta)
    del out
    if args.meta_out:
        os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)
        with open(args.meta_out, "w") as f:
            json.dump(meta, f, indent=2)
    print(f"profiled {args.op}: {meta}")


# --------------------------------------------------------------------------- #
# PLOT  (parse ncu CSV + meta -> SVG roofline)
# --------------------------------------------------------------------------- #
def _num(cell):
    if cell is None:
        return None
    try:
        return float(str(cell[0]).replace(",", "").strip())
    except ValueError:
        return None


def _to_seconds(cell):
    n = _num(cell)
    if n is None:
        return None
    u = (cell[1] or "").lower()
    if "nsecond" in u or u == "ns": return n * 1e-9
    if "usecond" in u or u == "us": return n * 1e-6
    if "msecond" in u or u == "ms": return n * 1e-3
    if "second" in u or u == "s":   return n
    return n * 1e-9


def _to_bytes(cell):
    n = _num(cell)
    if n is None:
        return None
    u = (cell[1] or "").lower()
    if "gbyte" in u: return n * 1e9
    if "mbyte" in u: return n * 1e6
    if "kbyte" in u: return n * 1e3
    return n


def parse_ncu_csv(path):
    groups = {}
    with open(path, newline="") as f:
        header = None
        for row in csv.reader(f):
            if header is None:
                if "Metric Name" in row and "Metric Value" in row:
                    header = row
                continue
            if not row or len(row) != len(header):
                continue
            d = dict(zip(header, row))
            kid = d.get("ID") or d.get("Kernel Name", "k")
            groups.setdefault(kid, {})[d["Metric Name"]] = (
                d.get("Metric Value"), d.get("Metric Unit", ""))
    return groups


def aggregate(groups):
    tot_t = tot_b = sm = dram = tens = 0.0
    have = {"sm": False, "dram": False, "tens": False}
    for metrics in groups.values():
        t = _to_seconds(metrics.get(M_TIME))
        if t is None:
            continue
        tot_t += t
        b = _to_bytes(metrics.get(M_BYTES))
        if b is not None:
            tot_b += b
        for key, mname in (("sm", M_SM), ("dram", M_DRAM), ("tens", M_TENSOR)):
            v = _num(metrics.get(mname))
            if v is not None:
                have[key] = True
                if key == "sm":   sm += v * t
                if key == "dram": dram += v * t
                if key == "tens": tens += v * t
    out = {"time": tot_t, "bytes": tot_b}
    if tot_t > 0:
        if have["sm"]:   out["sm_pct"] = sm / tot_t
        if have["dram"]: out["dram_pct"] = dram / tot_t
        if have["tens"]: out["tens_pct"] = tens / tot_t
    return out


def run_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pts = {}
    for meta_path in sorted(glob.glob(os.path.join(args.prof, "*.meta.json"))):
        stem = meta_path[:-len(".meta.json")]
        csv_path = stem + ".csv"
        if not os.path.exists(csv_path):
            print(f"skip {meta_path}: no {csv_path}")
            continue
        meta = json.load(open(meta_path))
        agg = aggregate(parse_ncu_csv(csv_path))
        if agg["time"] <= 0:
            print(f"skip {meta['op']}: no timing parsed (check metric names)")
            continue
        flops = meta.get("flops", 0)
        b = agg["bytes"] if agg["bytes"] > 0 else meta.get("bytes_model", 0)
        pts[meta["op"]] = dict(
            tflops=flops / agg["time"] / 1e12 if flops else 0.0,
            ai=(flops / b) if (flops and b) else None,
            bytes=b, time=agg["time"], degree_type=meta.get("degree_type", ""),
            sm=agg.get("sm_pct"), dram=agg.get("dram_pct"), tens=agg.get("tens_pct"))

    peak_tflops = args.peak_tflops
    bw_gbs = args.peak_bw
    if "gemm" in pts and pts["gemm"]["tflops"] > 0:
        peak_tflops = pts["gemm"]["tflops"]
    if "copy" in pts and pts["copy"]["bytes"] > 0:
        bw_gbs = pts["copy"]["bytes"] / pts["copy"]["time"] / 1e9
    bw_bps = bw_gbs * 1e9
    ridge = (peak_tflops * 1e12) / bw_bps

    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    ai = np.logspace(-1, 4, 300)
    ax.plot(ai, np.minimum(peak_tflops, bw_bps * ai / 1e12), color="black", lw=1.5)
    ax.axvline(ridge, color="gray", ls=":", lw=1)
    ax.text(ridge, peak_tflops * 1.05, f"ridge {ridge:.0f}", rotation=90,
            va="bottom", ha="right", fontsize=7, color="gray")

    if "gemm" in pts and pts["gemm"]["ai"]:
        g = pts["gemm"]
        ax.scatter([g["ai"]], [g["tflops"]], marker="o", s=60, color="#1b5e20",
                   edgecolor="black", lw=0.5, zorder=3, label="dense GEMM (fp16)")

    palette = {"regular": "#1565c0", "power-law": "#c62828"}
    fallback = ["#6a1b9a", "#00838f", "#ef6c00", "#4527a0"]
    fi = 0
    for op, p in pts.items():
        if op in ("gemm", "copy") or p["ai"] is None or p["tflops"] <= 0:
            continue
        col = palette.get(p["degree_type"])
        if col is None:
            col = fallback[fi % len(fallback)]; fi += 1
        marker = "^" if p["degree_type"] == "regular" else ("s" if p["degree_type"] == "power-law" else "D")
        ax.scatter([p["ai"]], [p["tflops"]], marker=marker, s=70, color=col,
                   edgecolor="black", lw=0.5, zorder=3, label=f"{op}  A·Aᵀ ({p['degree_type'] or 'n/a'})")
        ann = []
        if p["sm"] is not None:   ann.append(f"SM {p['sm']:.0f}%")
        if p["tens"] is not None: ann.append(f"T {p['tens']:.0f}%")
        if p["dram"] is not None: ann.append(f"DRAM {p['dram']:.0f}%")
        ax.annotate("  " + ", ".join(ann), (p["ai"], p["tflops"]),
                    fontsize=6.5, color=col, va="center")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity  (FLOP / DRAM byte, measured)")
    ax.set_ylabel("achieved throughput (TFLOPS)")
    ax.set_title("A·Aᵀ SpGEMM (real MTX datasets, fp32, cuSPARSE) on the empirical roofline")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, which="both", ls="-", lw=0.3, alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, format="svg")
    print(f"wrote {args.out}")
    print(f"ceilings: {peak_tflops:.0f} TFLOPS, {bw_gbs:.0f} GB/s, ridge {ridge:.0f}")
    for op, p in pts.items():
        if p["ai"] is not None:
            print(f"  {op:<14} AI={p['ai']:.3f}  {p['tflops']*1e3:.1f} GFLOPS  "
                  f"SM={p.get('sm')}  tensor={p.get('tens')}  DRAM={p.get('dram')}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("timing")
    pt.add_argument("--ms-root", default="MS", help="dir containing MS/<dataset>/ (from download_datasets.py --task spgemm)")
    pt.add_argument("--datasets", nargs="+", default=None, help="dataset names; default: auto-discover under --ms-root")
    pt.add_argument("--out", default="spgemm_mtx.csv")

    pp = sub.add_parser("profile")
    pp.add_argument("--op", required=True, help="dataset name (dir under --ms-root) OR anchor: gemm / copy")
    pp.add_argument("--ms-root", default="MS")
    pp.add_argument("--meta-out", default=None)

    pl = sub.add_parser("plot")
    pl.add_argument("--prof", required=True, help="dir of *.csv + *.meta.json")
    pl.add_argument("--out", default="spgemm_mtx_roofline.svg")
    pl.add_argument("--peak-tflops", type=float, default=835.0)   # H200 NVL bf16 dense
    pl.add_argument("--peak-bw", type=float, default=4800.0)      # GB/s, HBM3e

    args = ap.parse_args()
    {"timing": run_timing, "profile": run_profile, "plot": run_plot}[args.mode](args)


if __name__ == "__main__":
    main()
