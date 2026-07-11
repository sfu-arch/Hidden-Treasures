#!/usr/bin/env python3
"""
masked_spgemm_roofline.py  (v2)

Compares the GPU's execution modes for the two masked-SpGEMM primitives in
sparse attention against a dense baseline, and produces paper-grade evidence
that neither GPU mode is efficient:

    gemm              C = A @ B                       compute-bound reference
    spmm              C = S @ Bv                      attention context  P @ V
    sddmm_sampled     P = (A @ B) sampled at mask      attention scores   Q @ K^T
    sddmm_densemask   P = (A @ B), keep masked entries brute-force scores
    copy              streaming copy                   bandwidth anchor

Three modes:

  TIMING   wall-clock + self-calibrated roofline + crossover-density sweep.
           Supports structured masks so you can show the GPU underexploits
           even windowed/global/dilated attention sparsity, not just random.

  PROFILE  runs ONE op wrapped in an NVTX range "measure" and writes a meta
           sidecar (shapes, nnz, FLOPs). Drive it with Nsight Compute so the
           warmup kernels are excluded via --nvtx-include "measure/".

  PLOT     parses ncu CSV exports + meta sidecars and emits an SVG roofline
           scatter: each kernel placed at its MEASURED arithmetic intensity
           (FLOPs / actual DRAM bytes) vs achieved TFLOPS, annotated with
           %peak SM throughput and tensor-pipe utilization.

Examples
--------
# timing: crossover for a realistic sliding-window+global mask
python masked_spgemm_roofline.py timing --sweep --mask block_global --M 8192 --N 8192 --K 128

# profiling (run per op; see run_profile.sh for the full loop)
ncu --csv --log-file prof/sddmm_sampled.csv --nvtx --nvtx-include "measure/" \
    --metrics gpu__time_duration.sum,dram__bytes.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
    python masked_spgemm_roofline.py profile --op sddmm_sampled \
    --mask block_global --M 8192 --N 8192 --K 128 --meta-out prof/sddmm_sampled.meta.json

# plot
python masked_spgemm_roofline.py plot --prof prof/ --out roofline.svg
"""

import argparse
import csv
import glob
import json
import math
import os

# torch / matplotlib are imported lazily inside the modes that need them so that
# `plot` works on a machine without a GPU and `--help` works anywhere.

DENSE_DTYPE_NAME  = "bfloat16"   # tensor-core path for gemm / dense+mask
SPARSE_DTYPE_NAME = "float32"    # cuSPARSE SDDMM / SpMM path
ITERS   = 50
WARMUP  = 10

# ncu metric names (Hopper). These vary by ncu version/arch; centralized so you
# can edit in one place if a metric is reported under a different name.
M_TIME   = "gpu__time_duration.sum"
M_BYTES  = "dram__bytes.sum"
M_SM     = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
M_DRAM   = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
M_TENSOR = "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"
NCU_METRICS = ",".join([M_TIME, M_BYTES, M_SM, M_DRAM, M_TENSOR])

OPS = ["gemm", "spmm", "sddmm_sampled", "sddmm_densemask", "copy"]



# --------------------------------------------------------------------------- #
# mask families
# --------------------------------------------------------------------------- #
def build_mask(kind, M, N, density, dev, global_tokens=None, dilation=2):
    """Return a boolean (M, N) mask. Structured kinds approximate `density`;
    the actual density is reported by the caller."""
    import torch
    pos_i = (torch.arange(M, device=dev).float() * (N / M)).unsqueeze(1)  # (M,1)
    j     = torch.arange(N, device=dev).float().unsqueeze(0)              # (1,N)

    if kind == "random":
        mask = torch.rand(M, N, device=dev) < density

    elif kind == "block":  # sliding-window / local attention
        w = max(1, round(density * N))
        mask = (pos_i - j).abs() <= (w / 2.0)

    elif kind == "block_global":  # windowed local + a few global tokens
        g = global_tokens if global_tokens is not None else max(1, round(0.01 * N))
        g = min(g, M, N)
        global_nnz = g * N + g * M - g * g
        remaining = max(0.0, density * M * N - global_nnz)
        w = max(1, round(remaining / M))
        mask = (pos_i - j).abs() <= (w / 2.0)
        mask[:g, :] = True   # global query rows attend to everything
        mask[:, :g] = True   # everything attends to global key columns

    elif kind == "strided":  # dilated sliding window
        W = max(dilation, round(density * N * dilation / 2.0))
        diff = (pos_i - j)
        mask = (diff.abs() <= W) & (diff.round().long() % dilation == 0)

    else:
        raise ValueError(f"unknown mask kind: {kind}")

    return mask


def mask_to_csr(mask, dtype, dev):
    import torch
    M, N = mask.shape
    empty = ~mask.any(dim=1)
    if empty.any():
        cols = torch.randint(0, N, (int(empty.sum()),), device=dev)
        mask[empty.nonzero(as_tuple=True)[0], cols] = True
    dense = torch.zeros(M, N, device=dev, dtype=dtype)
    nnz = int(mask.sum())
    dense[mask] = torch.randn(nnz, device=dev, dtype=dtype)
    return dense.to_sparse_csr(), nnz, mask


# --------------------------------------------------------------------------- #
# operand construction (shared by timing + profiling)
# --------------------------------------------------------------------------- #
def build_op(op, M, K, N, density, mask_kind, dev, global_tokens=None, dilation=2):
    """Return (fn, meta). fn() runs the op once. meta has flops/bytes/nnz/etc."""
    import torch
    ddt = getattr(torch, DENSE_DTYPE_NAME)
    sdt = getattr(torch, SPARSE_DTYPE_NAME)
    meta = dict(op=op, M=M, K=K, N=N, mask=mask_kind, density_req=density)

    if op == "gemm":
        A = torch.randn(M, K, device=dev, dtype=ddt)
        B = torch.randn(K, N, device=dev, dtype=ddt)
        fn = lambda: torch.matmul(A, B)
        meta.update(flops=2 * M * N * K, nnz=M * N, dtype=DENSE_DTYPE_NAME,
                    bytes_model=2 * (M * K + K * N + M * N))

    elif op == "copy":
        n = 1 << 27
        src = torch.randn(n, device=dev, dtype=torch.float32)
        dst = torch.empty_like(src)
        fn = lambda: dst.copy_(src)
        meta.update(flops=0, nnz=0, dtype="float32", bytes_model=2 * n * 4)

    elif op in ("spmm", "sddmm_sampled", "sddmm_densemask"):
        mask = build_mask(mask_kind, M, N, density, dev, global_tokens, dilation)
        if op == "spmm":
            S_csr, Z, _ = mask_to_csr(mask, sdt, dev)
            Bv = torch.randn(N, K, device=dev, dtype=sdt)            # (N,K)
            fn = lambda: torch.sparse.mm(S_csr, Bv)                  # (M,K)
            meta.update(flops=2 * Z * K, nnz=Z, dtype=SPARSE_DTYPE_NAME)
        elif op == "sddmm_sampled":
            S_csr, Z, _ = mask_to_csr(mask, sdt, dev)
            Af = torch.randn(M, K, device=dev, dtype=sdt)
            Bf = torch.randn(K, N, device=dev, dtype=sdt)
            fn = lambda: torch.sparse.sampled_addmm(S_csr, Af, Bf, beta=0.0, alpha=1.0)
            meta.update(flops=2 * Z * K, nnz=Z, dtype=SPARSE_DTYPE_NAME)
        else:  # sddmm_densemask
            _, Z, m = mask_to_csr(mask, ddt, dev)
            A = torch.randn(M, K, device=dev, dtype=ddt)
            B = torch.randn(K, N, device=dev, dtype=ddt)
            fn = lambda: torch.matmul(A, B).masked_select(m)
            meta.update(flops=2 * M * N * K, nnz=Z, dtype=DENSE_DTYPE_NAME)
        meta["density_actual"] = meta["nnz"] / (M * N)
    else:
        raise ValueError(f"unknown op {op}")
    return fn, meta


# --------------------------------------------------------------------------- #
# TIMING mode
# --------------------------------------------------------------------------- #
def _time(fn, iters=ITERS, warmup=WARMUP):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        t.append(s.elapsed_time(e))
    t.sort()
    return t[len(t) // 2]


def calibrate(dev):
    import torch
    fg, _ = build_op("gemm", 8192, 8192, 8192, 0, "random", dev)
    ms = _time(fg)
    compute_tflops = (2 * 8192 ** 3) / (ms * 1e-3) / 1e12
    fc, mc = build_op("copy", 0, 0, 0, 0, "random", dev)
    ms = _time(fc)
    bw_gbs = mc["bytes_model"] / (ms * 1e-3) / 1e9
    ridge = (compute_tflops * 1e12) / (bw_gbs * 1e9)
    return compute_tflops, bw_gbs, ridge


def bench_point(M, K, N, density, mask_kind, dev, peak_tflops, args):
    import torch
    print(f"\n=== {mask_kind} mask  M={M} K={K} N={N}  req_density={density:.4f} ===")
    print(f"{'kernel':<26}{'ms':>9}{'TFLOPS':>10}{'%peak':>8}  note")
    print("-" * 74)
    res = {}
    for op in ["gemm", "spmm", "sddmm_sampled", "sddmm_densemask"]:
        try:
            fn, meta = build_op(op, M, K, N, density, mask_kind, dev,
                                args.global_tokens, args.dilation)
            ms = _time(fn)
            tflops = meta["flops"] / (ms * 1e-3) / 1e12
            pct = 100 * tflops / peak_tflops
            note = f"nnz={meta.get('nnz')}, dens={meta.get('density_actual', 1):.3f}"
            print(f"{op:<26}{ms:>9.3f}{tflops:>10.1f}{pct:>7.1f}%  {note}")
            res[op] = ms
        except Exception as ex:
            print(f"{op:<26}{'--':>9}{'--':>10}{'--':>8}  unsupported: {ex}")
    if "sddmm_sampled" in res and "sddmm_densemask" in res:
        win = "SAMPLED" if res["sddmm_sampled"] < res["sddmm_densemask"] else "+DENSEMASK"
        print(f"  -> SDDMM winner: {win}")
        return res["sddmm_sampled"] < res["sddmm_densemask"]
    return None


def run_timing(args):
    import torch
    assert torch.cuda.is_available(), "no CUDA device"
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    peak_tflops, bw_gbs, ridge = calibrate(dev)
    print(f"\n--- empirical roofline ---")
    print(f"compute ceiling : {peak_tflops:8.1f} TFLOPS")
    print(f"bandwidth       : {bw_gbs:8.1f} GB/s")
    print(f"ridge point     : {ridge:8.1f} flop/byte")

    if args.sweep:
        print(f"\n########## CROSSOVER SWEEP ({args.mask} mask) ##########")
        prev, crossover = None, None
        for d in [0.5, 0.25, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
            sampled_wins = bench_point(args.M, args.K, args.N, d, args.mask, dev,
                                       peak_tflops, args)
            if sampled_wins is not None:
                if prev is not None and sampled_wins and not prev:
                    crossover = d
                prev = sampled_wins
        if crossover:
            print(f"\nCrossover near density ~{crossover:.3%}: above it, the GPU's")
            print("fastest path is the dense product. That is the underutilization claim.")
        else:
            print("\nNo crossover in swept range (one strategy dominated throughout).")
    else:
        bench_point(args.M, args.K, args.N, args.density, args.mask, dev,
                    peak_tflops, args)


# --------------------------------------------------------------------------- #
# PROFILE mode  (run under ncu)
# --------------------------------------------------------------------------- #
def run_profile(args):
    import torch
    assert torch.cuda.is_available(), "no CUDA device"
    dev = "cuda"
    fn, meta = build_op(args.op, args.M, args.K, args.N, args.density, args.mask,
                        dev, args.global_tokens, args.dilation)
    for _ in range(3):          # warmup OUTSIDE the measured range
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("measure")   # ncu: --nvtx-include "measure"
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
# PLOT mode  (parse ncu CSV + meta, render SVG roofline)
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
    return n * 1e-9  # ncu time default is ns


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
    """Return {kernel_id: {metric_name: (value, unit)}} from an ncu --csv export."""
    groups = {}
    raw_lines = []
    with open(path, newline="") as f:
        header = None
        for row in csv.reader(f):
            raw_lines.append(row)
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
    if not groups:
        # Surface permission errors or other ncu failures from the raw output
        for row in raw_lines:
            text = " ".join(row)
            if "ERR_NVGPUCTRPERM" in text:
                raise RuntimeError(
                    f"{path}: GPU counter permission denied (ERR_NVGPUCTRPERM).\n"
                    "  Fix: sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'"
                )
            if "==ERROR==" in text:
                raise RuntimeError(f"{path}: ncu error: {text.strip()}")
    return groups


def aggregate(groups):
    """Sum time/bytes over a logical op's kernels; time-weight the % metrics."""
    tot_t = tot_b = 0.0
    sm = dram = tens = 0.0
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

    points = {}
    for meta_path in sorted(glob.glob(os.path.join(args.prof, "*.meta.json"))):
        stem = meta_path[:-len(".meta.json")]
        csv_path = stem + ".csv"
        if not os.path.exists(csv_path):
            print(f"skip {meta_path}: no matching {csv_path}")
            continue
        with open(meta_path) as f:
            print(meta_path)
            meta = json.load(f)
        agg = aggregate(parse_ncu_csv(csv_path))
        if agg["time"] <= 0:
            print(f"skip {meta['op']}: no timing parsed (check metric names)")
            continue
        flops = meta["flops"]
        b = agg["bytes"] if agg["bytes"] > 0 else meta.get("bytes_model", 0)
        points[meta["op"]] = dict(
            flops=flops, time=agg["time"], bytes=b,
            tflops=flops / agg["time"] / 1e12 if flops else 0.0,
            ai=(flops / b) if (flops and b) else None,
            sm=agg.get("sm_pct"), dram=agg.get("dram_pct"), tens=agg.get("tens_pct"),
            meta=meta)

    # ceilings: prefer measured anchors (gemm, copy), else CLI fallbacks
    peak_tflops = args.peak_tflops
    bw_gbs = args.peak_bw
    if "gemm" in points and points["gemm"]["tflops"] > 0:
        peak_tflops = points["gemm"]["tflops"]
    if "copy" in points and points["copy"]["bytes"] > 0 and points["copy"]["time"] > 0:
        bw_gbs = points["copy"]["bytes"] / points["copy"]["time"] / 1e9
    bw_bps = bw_gbs * 1e9
    ridge = (peak_tflops * 1e12) / bw_bps

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ai = np.logspace(-1, 4, 300)
    roof = np.minimum(peak_tflops, bw_bps * ai / 1e12)
    ax.plot(ai, roof, color="black", lw=1.5, zorder=1)
    ax.axvline(ridge, color="gray", ls=":", lw=1, zorder=0)
    ax.text(ridge, peak_tflops * 1.05, f"ridge {ridge:.0f}",
            rotation=90, va="bottom", ha="right", fontsize=7, color="gray")

    style = {
        "gemm":            ("o", "#1b5e20", "dense GEMM"),
        "sddmm_densemask": ("s", "#c62828", "SDDMM dense+mask"),
        "sddmm_sampled":   ("^", "#1565c0", "SDDMM sampled"),
        "spmm":            ("D", "#6a1b9a", "SpMM"),
    }
    for op, (mk, col, label) in style.items():
        p = points.get(op)
        if not p or p["ai"] is None or p["tflops"] <= 0:
            continue
        ax.scatter([p["ai"]], [p["tflops"]], marker=mk, s=70, color=col,
                   edgecolor="black", lw=0.5, zorder=3, label=label)
        ann = []
        if p["sm"] is not None:   ann.append(f"SM {p['sm']:.0f}%")
        if p["tens"] is not None: ann.append(f"T {p['tens']:.0f}%")
        if p["dram"] is not None: ann.append(f"DRAM {p['dram']:.0f}%")
        ax.annotate("  " + ", ".join(ann), (p["ai"], p["tflops"]),
                    fontsize=6.5, color=col, va="center")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity  (FLOP / DRAM byte, measured)")
    ax.set_ylabel("achieved throughput (TFLOPS)")
    ax.set_title("Masked-SpGEMM execution modes on the empirical roofline")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, which="both", ls="-", lw=0.3, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, format="svg")
    print(f"wrote {args.out}")
    print(f"ceilings: {peak_tflops:.0f} TFLOPS, {bw_gbs:.0f} GB/s, ridge {ridge:.0f} flop/byte")
    for op, p in points.items():
        if p["ai"] is not None:
            print(f"  {op:<18} AI={p['ai']:.2f}  {p['tflops']:.1f} TFLOPS  "
                  f"SM={p.get('sm')}  tensor={p.get('tens')}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def add_shape(p):
        p.add_argument("--M", type=int, default=8192)
        p.add_argument("--K", type=int, default=128)
        p.add_argument("--N", type=int, default=8192)
        p.add_argument("--density", type=float, default=0.05)
        p.add_argument("--mask", choices=["random", "block", "block_global", "strided"],
                       default="block_global")
        p.add_argument("--global-tokens", type=int, default=None)
        p.add_argument("--dilation", type=int, default=2)

    pt = sub.add_parser("timing", help="wall-clock + roofline + crossover sweep")
    add_shape(pt); pt.add_argument("--sweep", action="store_true")

    pp = sub.add_parser("profile", help="run ONE op under ncu (NVTX 'measure')")
    add_shape(pp)
    pp.add_argument("--op", choices=OPS, required=True)
    pp.add_argument("--meta-out", type=str, default=None)

    pl = sub.add_parser("plot", help="parse ncu CSV + meta -> SVG roofline")
    pl.add_argument("--prof", type=str, required=True, help="dir of *.csv + *.meta.json")
    pl.add_argument("--out", type=str, default="roofline.svg")
    pl.add_argument("--peak-tflops", type=float, default=835.0)  # H200 NVL bf16 dense
    pl.add_argument("--peak-bw", type=float, default=4800.0)     # GB/s, HBM3e

    args = ap.parse_args()
    {"timing": run_timing, "profile": run_profile, "plot": run_plot}[args.mode](args)


if __name__ == "__main__":
    main()