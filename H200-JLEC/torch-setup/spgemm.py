#!/usr/bin/env python3
"""
spgemm_roofline.py

Roofline profiler for  C = A @ A^T  (CSR x CSR -> CSR) on real graph
adjacencies, structured like masked_spgemm_roofline.py:

  timing   CUDA-event timing per dataset + self-calibrated roofline + CSV
  profile  run ONE op inside an NVTX "measure" range, write a meta sidecar
           (drive with Nsight Compute; see run_profile_spgemm.sh)
  plot     parse ncu CSV exports + meta -> SVG roofline scatter, each dataset
           placed at MEASURED arithmetic intensity (FLOP / DRAM byte) vs
           achieved TFLOPS, annotated with %peak SM and tensor-pipe util

Values are always computed in FP32. Graph adjacencies stored with int32 values
are converted to fp32 before the multiply, because cuSPARSE SpGEMM supports
float/double/complex only -- there is no integer path. (cuSPARSE SpGEMM also
requires 32-bit indices, which are enforced on load.) The point the roofline
makes: this fp32 SpGEMM runs on the CUDA cores, so it cannot touch the tensor
roof by construction, and it parks on the bandwidth roof.

Backend: CuPy's cusparseSpGEMM binding (cupyx.scipy.sparse csr @ csr).
Deps:    pip install cupy-cuda12x scipy numpy matplotlib   (match cupy to CUDA)

Ops for profile/plot: a dataset name (OGBN/RDT/AMZ/...) OR an anchor:
  gemm  large dense bf16 GEMM  -> empirical compute ceiling
  copy  streaming copy         -> empirical bandwidth ceiling
"""
import argparse
import csv
import glob
import json
import os
import numpy as np
import scipy.sparse as sp

WARMUP, REPEAT = 3, 10

M_TIME   = "gpu__time_duration.sum"
M_BYTES  = "dram__bytes.sum"
M_SM     = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
M_DRAM   = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
M_TENSOR = "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"
NCU_METRICS = ",".join([M_TIME, M_BYTES, M_SM, M_DRAM, M_TENSOR])


# --------------------------------------------------------------------------- #
# operands
# --------------------------------------------------------------------------- #
def load_adjacency_fp32(npz_path):
    """Load CSR adjacency; force int32 indices and FP32 values (upcast int32)."""
    A = sp.load_npz(npz_path).tocsr()
    A.indices = A.indices.astype(np.int32)
    A.indptr = A.indptr.astype(np.int32)
    A.data = A.data.astype(np.float32)          # int32 -> fp32 conversion here
    return A


def products_count(A_csr, At_csr):
    """Exact multiply-add pairs in A@A^T = sum over each nonzero (i,k) of A of
    nnz(row k of A^T). FLOP basis = 2 * products."""
    rowlen_At = np.diff(At_csr.indptr).astype(np.int64)
    return int(rowlen_At[A_csr.indices.astype(np.int64)].sum())


def build_op(op, indir, dev_gb=None):
    """Return (fn, meta). fn() runs the op once. Anchors: gemm, copy."""
    import cupy as cp
    import cupyx.scipy.sparse as csp

    if op == "gemm":                     # compute-ceiling anchor (bf16 tensor)
        n = 8192
        a = cp.random.rand(n, n, dtype=cp.float32).astype(cp.float16)
        b = cp.random.rand(n, n, dtype=cp.float32).astype(cp.float16)
        fn = lambda: a @ b
        return fn, dict(op="gemm", flops=2 * n ** 3, bytes_model=2 * 3 * n * n,
                        nnz_A=n * n, dtype="fp16")

    if op == "copy":                     # bandwidth-ceiling anchor
        nbytes = 1 << 30
        src = cp.empty(nbytes // 4, dtype=cp.float32)
        dst = cp.empty_like(src)
        fn = lambda: cp.copyto(dst, src)
        return fn, dict(op="copy", flops=0, bytes_model=2 * nbytes, dtype="fp32")

    # dataset op: A @ A^T in fp32
    path = os.path.join(indir, f"{op}.npz")
    A_cpu = load_adjacency_fp32(path)
    At_cpu = A_cpu.T.tocsr()
    At_cpu.indices = At_cpu.indices.astype(np.int32)
    At_cpu.indptr = At_cpu.indptr.astype(np.int32)
    prod = products_count(A_cpu, At_cpu)

    A = csp.csr_matrix((cp.asarray(A_cpu.data), cp.asarray(A_cpu.indices),
                        cp.asarray(A_cpu.indptr)), shape=A_cpu.shape)
    At = csp.csr_matrix((cp.asarray(At_cpu.data), cp.asarray(At_cpu.indices),
                         cp.asarray(At_cpu.indptr)), shape=At_cpu.shape)
    fn = lambda: A @ At
    meta = dict(op=op, flops=2 * prod, products=prod, nnz_A=int(A_cpu.nnz),
                rows=int(A_cpu.shape[0]), cols=int(A_cpu.shape[1]),
                dtype="fp32(from int32)")
    return fn, meta


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


def calibrate():
    fg, mg = build_op("gemm", ".")
    ms, _ = _time(fg)
    peak_tflops = mg["flops"] / (ms * 1e-3) / 1e12
    fc, mc = build_op("copy", ".")
    ms, _ = _time(fc)
    bw_gbs = mc["bytes_model"] / (ms * 1e-3) / 1e9
    return peak_tflops, bw_gbs, (peak_tflops * 1e12) / (bw_gbs * 1e9)


def run_timing(args):
    import cupy as cp
    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"device: {name}  cupy {cp.__version__}")
    peak_tflops, bw_gbs, ridge = calibrate()
    print(f"\n--- empirical roofline ---")
    print(f"compute ceiling : {peak_tflops:8.1f} TFLOPS (bf16 GEMM)")
    print(f"bandwidth       : {bw_gbs:8.1f} GB/s")
    print(f"ridge point     : {ridge:8.1f} flop/byte\n")

    fields = ["library", "dataset", "value_dtype", "rows", "nnz_A", "nnz_AAt",
              "products", "time_ms_median", "time_ms_min", "gflops"]
    rows = []
    print(f"{'dataset':>8}{'ms':>10}{'GFLOP/s':>10}   nnz_A -> nnz(AA^T)")
    print("-" * 60)
    for ds in args.datasets:
        path = os.path.join(args.indir, f"{ds}.npz")
        if not os.path.exists(path):
            print(f"{ds:>8}  skip (missing {path})")
            continue
        fn, meta = build_op(ds, args.indir)
        med, mn = _time(fn)
        C = fn(); nnz_c = int(C.nnz); del C
        cp.get_default_memory_pool().free_all_blocks()
        gfl = (meta["flops"] / 1e9) / (med / 1e3)
        print(f"{ds:>8}{med:>10.3f}{gfl:>10.2f}   {meta['nnz_A']:,} -> {nnz_c:,}")
        rows.append(dict(library="cuSPARSE", dataset=ds, value_dtype="FP32(from INT32)",
                         rows=meta["rows"], nnz_A=meta["nnz_A"], nnz_AAt=nnz_c,
                         products=meta["products"], time_ms_median=f"{med:.4f}",
                         time_ms_min=f"{mn:.4f}", gflops=f"{gfl:.2f}"))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.out}")


# --------------------------------------------------------------------------- #
# PROFILE  (run under ncu)
# --------------------------------------------------------------------------- #
def run_profile(args):
    import cupy as cp
    fn, meta = build_op(args.op, args.indir)
    for _ in range(WARMUP):
        fn()
    cp.cuda.Device().synchronize()
    cp.cuda.nvtx.RangePush("measure")        # ncu: --nvtx-include "measure/"
    out = fn()
    cp.cuda.Device().synchronize()
    cp.cuda.nvtx.RangePop()
    del out
    if args.meta_out:
        os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)
        with open(args.meta_out, "w") as f:
            json.dump(meta, f, indent=2)
    print(f"profiled {args.op}: {meta}")


# --------------------------------------------------------------------------- #
# PLOT  (parse ncu CSV + meta -> SVG roofline)   [mirrors masked profiler]
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
            bytes=b, time=agg["time"],
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
                   edgecolor="black", lw=0.5, zorder=3, label="dense GEMM (bf16)")

    palette = ["#1565c0", "#c62828", "#6a1b9a", "#00838f", "#ef6c00", "#4527a0"]
    di = 0
    for op, p in pts.items():
        if op in ("gemm", "copy") or p["ai"] is None or p["tflops"] <= 0:
            continue
        col = palette[di % len(palette)]; di += 1
        ax.scatter([p["ai"]], [p["tflops"]], marker="^", s=70, color=col,
                   edgecolor="black", lw=0.5, zorder=3, label=f"{op}  A·Aᵀ")
        ann = []
        if p["sm"] is not None:   ann.append(f"SM {p['sm']:.0f}%")
        if p["tens"] is not None: ann.append(f"T {p['tens']:.0f}%")
        if p["dram"] is not None: ann.append(f"DRAM {p['dram']:.0f}%")
        ax.annotate("  " + ", ".join(ann), (p["ai"], p["tflops"]),
                    fontsize=6.5, color=col, va="center")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity  (FLOP / DRAM byte, measured)")
    ax.set_ylabel("achieved throughput (TFLOPS)")
    ax.set_title("A·Aᵀ SpGEMM (fp32, cuSPARSE) on the empirical roofline")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, which="both", ls="-", lw=0.3, alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, format="svg")
    print(f"wrote {args.out}")
    print(f"ceilings: {peak_tflops:.0f} TFLOPS, {bw_gbs:.0f} GB/s, ridge {ridge:.0f}")
    for op, p in pts.items():
        if p["ai"] is not None:
            print(f"  {op:<8} AI={p['ai']:.3f}  {p['tflops']*1e3:.1f} GFLOPS  "
                  f"SM={p.get('sm')}  tensor={p.get('tens')}  DRAM={p.get('dram')}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("timing")
    pt.add_argument("--indir", default="data")
    pt.add_argument("--datasets", nargs="+", default=["OGBN", "RDT", "AMZ"])
    pt.add_argument("--out", default="cusparse_spgemm.csv")

    pp = sub.add_parser("profile")
    pp.add_argument("--op", required=True, help="dataset name OR gemm / copy")
    pp.add_argument("--indir", default="data")
    pp.add_argument("--meta-out", default=None)

    pl = sub.add_parser("plot")
    pl.add_argument("--prof", required=True, help="dir of *.csv + *.meta.json")
    pl.add_argument("--out", default="spgemm_roofline.svg")
    pl.add_argument("--peak-tflops", type=float, default=835.0)
    pl.add_argument("--peak-bw", type=float, default=4800.0)

    args = ap.parse_args()
    {"timing": run_timing, "profile": run_profile, "plot": run_plot}[args.mode](args)


if __name__ == "__main__":
    main()