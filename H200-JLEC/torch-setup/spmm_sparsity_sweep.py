#!/usr/bin/env python3
"""
spmm_sparsity_sweep.py

Benchmarks torch.sparse.mm (CSR sparse @ dense) across a grid of sparsity
levels and square problem sizes, converts wall-clock time into achieved
GFLOP/s (using the FLOPs actually required for the sparse matrix's nnz),
and renders a scatter plot of GFLOP/s vs. sparsity, one series per size.

Example
-------
python spmm_sparsity_sweep.py --out spmm_sparsity.svg
python spmm_sparsity_sweep.py --sizes 4096 8192 --densities 0.001 0.01 0.05 --iters 30
"""

import argparse

DENSITIES_PCT = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0]   # percent
SIZES = [4096, 8192, 16384]                       # M = N = K
DTYPE_NAME = "float32"                             # cuSPARSE SpMM path
ITERS = 30
WARMUP = 10


def build_spmm(M, N, K, density, dtype, dev):
    """Return (fn, nnz). fn() computes C = A_csr @ B once. A is (M,K), B is (K,N)."""
    import torch
    mask = torch.rand(M, K, device=dev) < density
    empty = ~mask.any(dim=1)
    if empty.any():
        cols = torch.randint(0, K, (int(empty.sum()),), device=dev)
        mask[empty.nonzero(as_tuple=True)[0], cols] = True
    nnz = int(mask.sum())
    dense = torch.zeros(M, K, device=dev, dtype=dtype)
    dense[mask] = torch.randn(nnz, device=dev, dtype=dtype)
    A_csr = dense.to_sparse_csr()
    B = torch.randn(K, N, device=dev, dtype=dtype)
    fn = lambda: torch.sparse.mm(A_csr, B)
    return fn, nnz


def time_op(fn, dev, iters, warmup):
    import torch
    for _ in range(warmup):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1e-3)  # ms -> s
    else:
        import time
        times = []
        for _ in range(iters):
            t0 = time.perf_counter(); fn(); t1 = time.perf_counter()
            times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


def run_sweep(sizes, densities_pct, iters, warmup):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, DTYPE_NAME)
    print(f"device: {dev}" + (f"  ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    print(f"{'M=N=K':>8}{'density%':>10}{'nnz':>12}{'time(ms)':>12}{'GFLOP/s':>12}")
    print("-" * 56)

    results = []  # list of dicts: size, density_pct, gflops, nnz, time_s
    for size in sizes:
        M = N = K = size
        for dpct in densities_pct:
            density = dpct / 100.0
            fn, nnz = build_spmm(M, N, K, density, dtype, dev)
            t = time_op(fn, dev, iters, warmup)
            flops = 2 * nnz * N
            gflops = flops / t / 1e9
            print(f"{size:>8}{dpct:>10.2f}{nnz:>12}{t*1e3:>12.3f}{gflops:>12.1f}")
            results.append(dict(size=size, density_pct=dpct, nnz=nnz, time_s=t, gflops=gflops))
    return results


def plot_results(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted(set(r["size"] for r in results))
    colors = {4096: "#1565c0", 8192: "#2e7d32", 16384: "#c62828"}
    markers = {4096: "o", 8192: "^", 16384: "s"}
    fallback_colors = ["#1565c0", "#2e7d32", "#c62828", "#6a1b9a", "#ef6c00"]
    fallback_markers = ["o", "^", "s", "D", "v"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, size in enumerate(sizes):
        col = colors.get(size, fallback_colors[i % len(fallback_colors)])
        mk = markers.get(size, fallback_markers[i % len(fallback_markers)])
        xs = [r["density_pct"] for r in results if r["size"] == size]
        ys = [r["gflops"] for r in results if r["size"] == size]
        ax.scatter(xs, ys, color=col, marker=mk, s=80, edgecolor="black", lw=0.5,
                   label=f"M=N=K={size}", zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("sparsity (density, %)")
    ax.set_ylabel("achieved throughput (GFLOP/s)")
    ax.set_title("SpMM (torch.sparse.mm) throughput vs. sparsity")
    ax.grid(True, which="both", ls="-", lw=0.3, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES,
                    help="square problem sizes (M=N=K)")
    ap.add_argument("--densities", type=float, nargs="+", default=DENSITIES_PCT,
                    help="sparsity levels as percent (e.g. 0.1 1 3)")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--out", type=str, default="spmm_sparsity.png",
                    help="output plot path (.png or .svg)")
    args = ap.parse_args()

    results = run_sweep(args.sizes, args.densities, args.iters, args.warmup)
    plot_results(results, args.out)


if __name__ == "__main__":
    main()
