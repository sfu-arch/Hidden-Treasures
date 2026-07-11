#!/usr/bin/env python3
"""
download_datasets.py

Fetch the three graph adjacencies and save each as a scipy CSR .npz for the
cuSPARSE profiler:

    OGBN -> ogbn-products        (OGB)
    RDT  -> Reddit               (PyG)
    AMZ  -> AmazonProducts       (PyG, the GraphSAINT amazon graph)

Each adjacency A is symmetrized (graphs may be directed), given unit weights
(structure only), and stored with int32 indptr/indices -- cuSPARSE SpGEMM only
supports 32-bit indices.

Deps:  pip install ogb torch-geometric scipy numpy
       (torch-geometric must match your installed torch / CUDA build)

Usage: python download_datasets.py --datasets OGBN RDT AMZ --outdir data
       # swap AMZ to a different Amazon graph by editing load_amazon().
"""
import argparse
import os
import numpy as np
import scipy.sparse as sp


def _to_symmetric_csr(rows, cols, n):
    """Structure-only, symmetrized adjacency as CSR with int32 indices."""
    data = np.ones(rows.shape[0], dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    A = (A + A.T).tocsr()          # symmetrize directed edges
    A.sum_duplicates()
    A.data = np.ones_like(A.data, dtype=np.float32)   # collapse to structure
    A.indices = A.indices.astype(np.int32)
    A.indptr = A.indptr.astype(np.int32)
    return A


def load_ogbn_products(outdir):
    from ogb.nodeproppred import NodePropPredDataset
    ds = NodePropPredDataset(name="ogbn-products", root=os.path.join(outdir, "ogb"))
    graph, _ = ds[0]
    ei = graph["edge_index"]                       # (2, E), numpy
    n = int(graph["num_nodes"])
    return _to_symmetric_csr(ei[0].astype(np.int64), ei[1].astype(np.int64), n)


def load_reddit(outdir):
    from torch_geometric.datasets import Reddit
    ds = Reddit(root=os.path.join(outdir, "Reddit"))
    d = ds[0]
    ei = d.edge_index.numpy()
    return _to_symmetric_csr(ei[0].astype(np.int64), ei[1].astype(np.int64), int(d.num_nodes))


def load_amazon(outdir):
    # GraphSAINT AmazonProducts (~1.6M nodes, ~132M edges). If you meant a
    # different "Amazon" (e.g. SNAP com-Amazon, or ogbn-... ), swap this loader.
    from torch_geometric.datasets import AmazonProducts
    ds = AmazonProducts(root=os.path.join(outdir, "AmazonProducts"))
    d = ds[0]
    ei = d.edge_index.numpy()
    return _to_symmetric_csr(ei[0].astype(np.int64), ei[1].astype(np.int64), int(d.num_nodes))


LOADERS = {"OGBN": load_ogbn_products, "RDT": load_reddit, "AMZ": load_amazon}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(LOADERS),
                    choices=list(LOADERS))
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for name in args.datasets:
        print(f"\n=== {name} ===")
        try:
            A = LOADERS[name](args.outdir)
        except ImportError as e:
            print(f"  missing dependency for {name}: {e}\n  -> pip install ogb torch-geometric")
            continue
        except Exception as e:
            print(f"  failed to load {name}: {e}")
            continue
        path = os.path.join(args.outdir, f"{name}.npz")
        sp.save_npz(path, A)
        density = A.nnz / (A.shape[0] * A.shape[1])
        print(f"  saved {path}: {A.shape[0]:,} x {A.shape[1]:,}  "
              f"nnz={A.nnz:,}  density={density:.2e}  avg_deg={A.nnz / A.shape[0]:.1f}")


if __name__ == "__main__":
    main()