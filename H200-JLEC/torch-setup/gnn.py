#!/usr/bin/env python3
"""
gnn.py

Real-dataset counterpart to spmm_mtx.py for the GNN datasets fetched by:

    python download_datasets.py --task gnn --all

Benchmarks the core GNN message-passing primitive -- SpMM: H' = A @ H, where
A is the (symmetrized, structure-only) graph adjacency and H is a dense
node-feature matrix -- via torch.sparse.mm (same backend as spmm_mtx.py).

download_datasets.py routes each GNN dataset to HS/<name>/ (Highly Sparse) or
MS/<name>/ (Moderately Sparse: ogbn-proteins, Reddit, Amazon-Photo,
Amazon-Computers, Cora) and saves it in its native PyG/OGB on-disk format
(raw/processed .pt files, not edges.npy/.mtx like the SpGEMM loader). This
script reloads each dataset through the same PyG/OGB dataset class
download_datasets.py used to fetch it -- no network access needed once the
files are cached on disk. A couple of those classes name their on-disk
folder differently than the registry name download_datasets.py renamed it
to (e.g. PyG's Amazon("Computers") vs. the "Amazon-Computers" directory, or
OGB's "ogbn_arxiv" vs. the "ogbn-arxiv" directory); for those, the existing
directory is symlinked under the name the class expects rather than copied.

Modes (mirrors spmm_mtx.py / spgemm.py / masked_spgemm_roofline.py):
  timing   CUDA-event timing per dataset + self-calibrated roofline -> CSV
  profile  run ONE dataset inside an NVTX "measure" range, write a meta
           sidecar (drive with Nsight Compute; see run_profile.sh)
  plot     parse ncu CSV exports + meta -> SVG roofline scatter, each dataset
           placed at MEASURED arithmetic intensity (FLOP / DRAM byte) vs
           achieved TFLOPS, annotated with %peak SM and tensor-pipe util

Deps: pip install torch-geometric ogb   (only needed for datasets that use them)

Examples
--------
python gnn.py timing --root . --all --F 128
python gnn.py timing --root . --datasets Cora Reddit --out gnn.csv

python gnn.py profile --op Cora --root . --F 128 --meta-out prof_gnn/Cora.meta.json
python gnn.py plot --prof prof_gnn --out gnn_roofline.svg
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

WARMUP, ITERS = 10, 30
DTYPE_NAME = "float32"   # cuSPARSE SpMM path (matches spmm_sparsity_sweep.py)

M_TIME   = "gpu__time_duration.sum"
M_BYTES  = "dram__bytes.sum"
M_SM     = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
M_DRAM   = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
M_TENSOR = "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"
NCU_METRICS = ",".join([M_TIME, M_BYTES, M_SM, M_DRAM, M_TENSOR])

# download_datasets.py lives alongside this script.
sys.path.insert(0, str(Path(__file__).parent))


# --------------------------------------------------------------------------- #
# dataset loading — reload the already-downloaded PyG/OGB dataset in place
# --------------------------------------------------------------------------- #
def _reload_dataset(name: str, loader_key: str, dataset_dir: Path):
    """Return the PyG Data object for an already-downloaded GNN dataset,
    reloading it through the same class download_datasets.py used to fetch
    it (no re-download if raw/processed files are present)."""
    if loader_key == "pyg_reddit":
        from torch_geometric.datasets import Reddit
        return Reddit(root=str(dataset_dir))[0]
    if loader_key == "pyg_flickr":
        from torch_geometric.datasets import Flickr
        return Flickr(root=str(dataset_dir))[0]
    if loader_key == "pyg_planetoid":
        from torch_geometric.datasets import Planetoid
        return Planetoid(root=str(dataset_dir.parent), name=name)[0]

    # Amazon / Coauthor / OGB: download_datasets.py renamed the on-disk dir
    # from the class's own subset/dir name to the registry name (e.g.
    # "Amazon-Computers" instead of PyG's "Computers", "ogbn-arxiv" instead
    # of OGB's "ogbn_arxiv"). Symlink the existing dir under the name the
    # class expects instead of duplicating data on disk.
    if loader_key == "pyg_amazon":
        from torch_geometric.datasets import Amazon
        subset = name.split("-", 1)[1]
        cls, kwargs, expected = Amazon, dict(name=subset), subset
    elif loader_key == "pyg_coauthor":
        from torch_geometric.datasets import Coauthor
        subset = name.split("-", 1)[1]
        cls, kwargs, expected = Coauthor, dict(name=subset), subset
    elif loader_key == "ogb":
        from ogb.nodeproppred import PygNodePropPredDataset
        cls, kwargs, expected = PygNodePropPredDataset, dict(name=name), name.replace("-", "_")
    else:
        raise ValueError(f"unknown loader_key {loader_key!r} for dataset {name!r}")

    tmp_root = Path(tempfile.mkdtemp(prefix="gnn_reload_"))
    try:
        (tmp_root / expected).symlink_to(dataset_dir.resolve())
        dataset = cls(root=str(tmp_root), **kwargs)
        return dataset[0]
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def load_dataset(name: str, root: Path):
    """Return (edge_index[2,E] numpy, num_nodes, category) for GNN dataset `name`."""
    import download_datasets as dd

    if name not in dd.GNN_DATASETS:
        raise FileNotFoundError(
            f"unknown GNN dataset '{name}'. See: python download_datasets.py --task gnn --list"
        )
    loader_key, _ = dd.GNN_DATASETS[name]
    category = "MS" if name in dd.GNN_MS_DATASETS else "HS"
    dataset_dir = root / category / name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"'{dataset_dir}' not found. Fetch it with:\n"
            f"  python download_datasets.py --task gnn --datasets {name}"
        )

    data = _reload_dataset(name, loader_key, dataset_dir)
    edge_index = data.edge_index.numpy()
    num_nodes = int(data.num_nodes)
    return edge_index, num_nodes, category


def discover_datasets(root: Path):
    import download_datasets as dd
    names = []
    for name in dd.GNN_DATASETS:
        category = "MS" if name in dd.GNN_MS_DATASETS else "HS"
        if (root / category / name).is_dir():
            names.append(name)
    return sorted(names)


# --------------------------------------------------------------------------- #
# operands
# --------------------------------------------------------------------------- #
def build_op(op, root, F, dev):
    """Return (fn, meta). fn() computes H' = A_csr @ H once. Anchors: gemm, copy."""
    import torch

    if op == "gemm":                     # compute-ceiling anchor (bf16 tensor)
        n = 8192
        a = torch.randn(n, n, device=dev, dtype=torch.float32)
        b = torch.randn(n, n, device=dev, dtype=torch.float32)
        fn = lambda: torch.matmul(a, b)
        return fn, dict(op="gemm", flops=2 * n ** 3, bytes_model=2 * 3 * n * n,
                        nnz=n * n, dtype="float32")

    if op == "copy":                     # bandwidth-ceiling anchor
        n = 1 << 27
        src = torch.randn(n, device=dev, dtype=torch.float32)
        dst = torch.empty_like(src)
        fn = lambda: dst.copy_(src)
        return fn, dict(op="copy", flops=0, bytes_model=2 * n * 4, nnz=0, dtype="float32")

    edge_index, num_nodes, category = load_dataset(op, root)
    dtype = getattr(torch, DTYPE_NAME)

    import scipy.sparse as sp
    rows = np.concatenate([edge_index[0], edge_index[1]]).astype(np.int64)
    cols = np.concatenate([edge_index[1], edge_index[0]]).astype(np.int64)   # symmetrize
    vals = np.ones(len(rows), dtype=np.float32)
    A_csr = sp.coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    A_csr.sum_duplicates()
    A_csr.data[:] = 1.0                                                     # structure-only
    nnz = int(A_csr.nnz)

    A = torch.sparse_csr_tensor(
        torch.from_numpy(A_csr.indptr.astype(np.int64)),
        torch.from_numpy(A_csr.indices.astype(np.int64)),
        torch.from_numpy(A_csr.data.astype(np.float32)),
        size=(num_nodes, num_nodes), device=dev, dtype=dtype,
    )
    H = torch.randn(num_nodes, F, device=dev, dtype=dtype)
    fn = lambda: torch.sparse.mm(A, H)

    out_meta = dict(op=op, flops=2 * nnz * F, nnz=nnz, rows=num_nodes, cols=num_nodes,
                    F=F, dtype=DTYPE_NAME, bytes_model=4 * (nnz + 2 * num_nodes * F),
                    category=category)
    return fn, out_meta


# --------------------------------------------------------------------------- #
# TIMING
# --------------------------------------------------------------------------- #
def _time(fn):
    import torch
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(ITERS):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))   # ms
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def calibrate(root, dev):
    import torch
    fg, mg = build_op("gemm", root, 0, dev)
    ms, _ = _time(fg)
    peak_tflops = mg["flops"] / (ms * 1e-3) / 1e12
    fc, mc = build_op("copy", root, 0, dev)
    ms, _ = _time(fc)
    bw_gbs = mc["bytes_model"] / (ms * 1e-3) / 1e9
    return peak_tflops, bw_gbs, (peak_tflops * 1e12) / (bw_gbs * 1e9)


def run_timing(args):
    import torch
    assert torch.cuda.is_available(), "no CUDA device"
    dev = "cuda"
    root = Path(args.root)
    datasets = args.datasets or discover_datasets(root)
    if not datasets:
        raise SystemExit(f"no GNN datasets found under '{root}' and none passed via --datasets")

    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    peak_tflops, bw_gbs, ridge = calibrate(root, dev)
    print(f"\n--- empirical roofline ---")
    print(f"compute ceiling : {peak_tflops:8.1f} TFLOPS (bf16 GEMM)")
    print(f"bandwidth       : {bw_gbs:8.1f} GB/s")
    print(f"ridge point     : {ridge:8.1f} flop/byte\n")

    fields = ["dataset", "rows", "nnz", "F", "time_ms_median", "time_ms_min",
              "gflops", "pct_peak", "category"]
    out_rows = []
    print(f"{'dataset':>18}{'rows':>10}{'nnz':>12}{'ms':>10}{'GFLOP/s':>10}{'%peak':>8}")
    print("-" * 72)
    for ds in datasets:
        try:
            fn, meta = build_op(ds, root, args.F, dev)
        except FileNotFoundError as exc:
            print(f"{ds:>18}  skip ({exc})")
            continue
        med, mn = _time(fn)
        gfl = (meta["flops"] / 1e9) / (med / 1e3)
        pct = 100 * (gfl / 1e3) / peak_tflops
        print(f"{ds:>18}{meta['rows']:>10,}{meta['nnz']:>12,}{med:>10.3f}{gfl:>10.2f}{pct:>7.1f}%")
        out_rows.append(dict(dataset=ds, rows=meta["rows"], nnz=meta["nnz"],
                             F=args.F, time_ms_median=f"{med:.4f}", time_ms_min=f"{mn:.4f}",
                             gflops=f"{gfl:.2f}", pct_peak=f"{pct:.2f}", category=meta["category"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {args.out}")


# --------------------------------------------------------------------------- #
# PROFILE  (run under ncu)
# --------------------------------------------------------------------------- #
def run_profile(args):
    import torch
    assert torch.cuda.is_available(), "no CUDA device"
    dev = "cuda"
    root = Path(args.root)
    fn, meta = build_op(args.op, root, args.F, dev)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("measure")        # ncu: --nvtx-include "measure/"
    out = fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
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
            bytes=b, time=agg["time"], category=meta.get("category", ""),
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

    palette = {"MS": "#1565c0", "HS": "#c62828"}
    fallback = ["#6a1b9a", "#00838f", "#ef6c00", "#4527a0"]
    fi = 0
    for op, p in pts.items():
        if op in ("gemm", "copy") or p["ai"] is None or p["tflops"] <= 0:
            continue
        col = palette.get(p["category"])
        if col is None:
            col = fallback[fi % len(fallback)]; fi += 1
        marker = "^" if p["category"] == "MS" else ("s" if p["category"] == "HS" else "D")
        ax.scatter([p["ai"]], [p["tflops"]], marker=marker, s=70, color=col,
                   edgecolor="black", lw=0.5, zorder=3, label=f"{op}  A@H ({p['category'] or 'n/a'})")
        ann = []
        if p["sm"] is not None:   ann.append(f"SM {p['sm']:.0f}%")
        if p["tens"] is not None: ann.append(f"T {p['tens']:.0f}%")
        if p["dram"] is not None: ann.append(f"DRAM {p['dram']:.0f}%")
        ax.annotate("  " + ", ".join(ann), (p["ai"], p["tflops"]),
                    fontsize=6.5, color=col, va="center")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity  (FLOP / DRAM byte, measured)")
    ax.set_ylabel("achieved throughput (TFLOPS)")
    ax.set_title("GNN message passing (A@H, torch.sparse.mm) on the empirical roofline")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, which="both", ls="-", lw=0.3, alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, format="svg")
    print(f"wrote {args.out}")
    print(f"ceilings: {peak_tflops:.0f} TFLOPS, {bw_gbs:.0f} GB/s, ridge {ridge:.0f}")
    for op, p in pts.items():
        if p["ai"] is not None:
            print(f"  {op:<18} AI={p['ai']:.3f}  {p['tflops']*1e3:.1f} GFLOPS  "
                  f"SM={p.get('sm')}  tensor={p.get('tens')}  DRAM={p.get('dram')}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("timing")
    pt.add_argument("--root", default=".", help="dir containing HS/ and MS/ (from download_datasets.py --task gnn)")
    pt.add_argument("--datasets", nargs="+", default=None, help="dataset names; default: auto-discover under --root")
    pt.add_argument("--F", type=int, default=256, help="dense node-feature width H=(num_nodes,F)")
    pt.add_argument("--out", default="gnn.csv")

    pp = sub.add_parser("profile")
    pp.add_argument("--op", required=True, help="dataset name (under HS/ or MS/) OR anchor: gemm / copy")
    pp.add_argument("--root", default=".")
    pp.add_argument("--F", type=int, default=128)
    pp.add_argument("--meta-out", default=None)

    pl = sub.add_parser("plot")
    pl.add_argument("--prof", required=True, help="dir of *.csv + *.meta.json")
    pl.add_argument("--out", default="gnn_roofline.svg")
    pl.add_argument("--peak-tflops", type=float, default=835.0)   # H200 NVL bf16 dense
    pl.add_argument("--peak-bw", type=float, default=4800.0)      # GB/s, HBM3e

    args = ap.parse_args()
    {"timing": run_timing, "profile": run_profile, "plot": run_plot}[args.mode](args)


if __name__ == "__main__":
    main()
