"""
Download real-world graph datasets for GNN or Transitive Closure workload simulation.

Datasets are organised into two sparsity categories:
  HS (Highly Sparse)      → ./HS/
  MS (Moderately Sparse)  → ./MS/

Routing rules:
  --task tc     → all datasets → HS
  --task spgemm → all datasets → MS
  --task gnn    → ogbn-proteins, Reddit, Amazon-Photo, Amazon-Computers, Cora → MS
                  all other GNN datasets → HS

Usage:
  # ── GNN datasets (split between ./HS and ./MS) ─────────────────────────────
  python download_datasets.py --task gnn --all
  python download_datasets.py --task gnn --datasets ogbn-arxiv Reddit Cora
  python download_datasets.py --task gnn --list

  # ── Transitive Closure datasets (saved under ./HS) ────────
  python download_datasets.py --task tc --all
  python download_datasets.py --task tc --datasets cit-HepPh soc-Epinions1
  python download_datasets.py --task tc --list

  # ── SpGEMM datasets (saved under ./MS) ─────────────────────────────────
  python download_datasets.py --task spgemm --all
  python download_datasets.py --task spgemm --datasets poisson3Da circuit_2
  python download_datasets.py --task spgemm --list

GNN datasets (≤250K nodes):
  ogbn-proteins  ~132K nodes, ~39.6M edges
  ogbn-arxiv     ~169K nodes, ~1.2M edges
  Reddit         ~233K nodes, ~114M edges
  Flickr          ~89K nodes, ~899K edges
  Cora             ~2.7K nodes, ~10.5K edges
  CiteSeer         ~3.3K nodes,  ~9.1K edges
  PubMed          ~19.7K nodes,  ~88K edges
  Amazon-Computers ~13.7K nodes, ~491K edges
  Amazon-Photo      ~7.6K nodes, ~238K edges
  Coauthor-CS      ~18.3K nodes, ~163K edges
  Coauthor-Physics ~34.5K nodes, ~495K edges

SpGEMM (Moderately Sparse × Moderately Sparse) datasets — density [0.001, 0.01], ≤250K nodes:
  Fluid Dynamics — Regular (low degree STD, structured FEM/CFD mesh):
    poisson3Da   ~13.5K nodes, ~352K  NNZ, density=1.93e-3 — 3D Poisson CFD (COMSOL)
    cavity16     ~ 4.6K nodes, ~137K  NNZ, density=6.62e-3 — driven cavity CFD
    Goodwin_040  ~17.9K nodes, ~561K  NNZ, density=1.75e-3 — Navier–Stokes FEM
    rdb5000      ~ 5.0K nodes, ~ 29K  NNZ, density=1.18e-3 — reaction-diffusion CFD
  Fluid Dynamics / Circuit — Power-Law (high degree STD, irregular topology):
    sme3Db       ~29.1K nodes, ~2.1M  NNZ, density=2.46e-3 — 3D structural mech FEM
    circuit_2    ~ 4.5K nodes, ~ 21K  NNZ, density=1.04e-3 — circuit DAE simulation
    msc10848     ~10.8K nodes, ~1.2M  NNZ, density=1.04e-2 — structural assembly FEM

Transitive Closure datasets (≤250K nodes):
  Regular (low degree STD — near-DAG / hierarchical structure):
    GeneOntology    ~47K  nodes, ~75K  edges  — biological process DAG (STD ~1.6)
    cit-HepTh       ~27K  nodes, ~352K edges  — citation DAG         (STD ~12)
    cit-HepPh       ~34K  nodes, ~421K edges  — citation DAG         (STD ~15)

  Power-Law (high degree STD — directed social / web):
    email-Enron     ~36K  nodes, ~183K edges  — email graph          (STD ~80)
    soc-Epinions1   ~75K  nodes, ~508K edges  — trust social network (STD ~130)
    soc-Slashdot0811 ~77K nodes, ~905K edges  — Slashdot social      (STD ~160)
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# GNN Dataset registry
# Each entry:  name -> (loader_key, approximate_num_nodes)
# ---------------------------------------------------------------------------

GNN_DATASETS: dict[str, tuple] = {
    # OGB datasets (require `pip install ogb`)
    "ogbn-proteins": ("ogb",          132534),
    "ogbn-arxiv":    ("ogb",          169343),
    # PyG native
    "Reddit":              ("pyg_reddit",   232965),
    "Flickr":              ("pyg_flickr",    89250),
    "Cora":                ("pyg_planetoid",  2708),
    "CiteSeer":            ("pyg_planetoid",  3327),
    "PubMed":              ("pyg_planetoid", 19717),
    "Amazon-Computers":    ("pyg_amazon",    13752),
    "Amazon-Photo":        ("pyg_amazon",     7650),
    "Coauthor-CS":         ("pyg_coauthor",  18333),
    "Coauthor-Physics":    ("pyg_coauthor",  34493),
}

# Keep old name as alias for backward compatibility
DATASETS = GNN_DATASETS

# GNN datasets that belong to the Moderately Sparse (MS) category.
# All others go to the Highly Sparse (HS) category.
GNN_MS_DATASETS: frozenset[str] = frozenset({
    "ogbn-proteins",
    "Reddit",
    "Amazon-Photo",
    "Amazon-Computers",
    "Cora",
})

# ---------------------------------------------------------------------------
# Transitive Closure Dataset registry
# Each entry:  name -> (loader_key, approx_nodes, approx_edges, degree_type)
#   degree_type: "regular" (low STD) or "power-law" (high STD)
# ---------------------------------------------------------------------------

TC_DATASETS: dict[str, tuple] = {
    # ── Regular: near-DAG / hierarchical, low degree STD ───────────────────
    # Directed acyclic graph of biological process terms.
    # Source: http://purl.obolibrary.org/obo/go/go-basic.obo
    "GeneOntology":      ("tc_go",       47284,   75834,  "regular"),
    # High Energy Physics — Theory citation network (directed DAG by time).
    # Source: https://snap.stanford.edu/data/cit-HepTh.html
    "cit-HepTh":         ("tc_snap",     27770,  352807,  "regular"),
    # High Energy Physics — Phenomenology citation network (directed DAG).
    # Source: https://snap.stanford.edu/data/cit-HepPh.html
    "cit-HepPh":         ("tc_snap",     34546,  421578,  "regular"),

    # ── Power-Law: directed social / communication, high degree STD ─────────
    # Email communication network from Enron corpus.
    # Source: https://snap.stanford.edu/data/email-Enron.html
    "email-Enron":       ("tc_snap",     36692,  183831,  "power-law"),
    # Epinions directed "who-trusts-whom" social network.
    # Source: https://snap.stanford.edu/data/soc-Epinions1.html
    "soc-Epinions1":     ("tc_snap",     75879,  508837,  "power-law"),
    # Slashdot social network (friend/foe links, Aug 2008).
    # Source: https://snap.stanford.edu/data/soc-Slashdot0811.html
    "soc-Slashdot0811":  ("tc_snap",     77360,  905468,  "power-law"),
}

# ---------------------------------------------------------------------------
# SpGEMM Dataset registry  (Moderately Sparse × Moderately Sparse, ILU SpGEMM)
# Each entry:  name -> (loader_key, group, approx_nodes, approx_nnz, density, degree_type, domain)
# ---------------------------------------------------------------------------

SPGEMM_DATASETS: dict[str, tuple] = {
    # ── Regular: structured FEM/CFD mesh, low degree STD ───────────────────
    # 3D Poisson problem discretized by COMSOL/FEMLAB.
    # 26 NNZ/row (structured 3D stencil) → very uniform.
    "poisson3Da":  ("spgemm_ss", "FEMLAB",  13514,   352762, 1.93e-3, "regular",    "Computational Fluid Dynamics"),
    # Driven cavity (Re=0) Navier-Stokes FDM on 20×20 grid.
    # ~30 NNZ/row, structured stencil.
    "cavity16":    ("spgemm_ss", "DRIVCAV",  4562,   137887, 6.62e-3, "regular",    "Computational Fluid Dynamics"),
    # Navier-Stokes + transport FEM (small mesh).
    # ~31 NNZ/row, structured FEM pattern.
    "Goodwin_040": ("spgemm_ss", "Goodwin", 17922,   561677, 1.75e-3, "regular",    "Computational Fluid Dynamics"),
    # Reaction-diffusion Brusselator model (2D structured grid).
    # ~6 NNZ/row (PDE stencil) — most regular pattern.
    "rdb5000":     ("spgemm_ss", "Bai",      5000,    29600, 1.18e-3, "regular",    "Computational Fluid Dynamics"),
    # ── Power-Law: irregular topology, high degree STD ──────────────────────
    # 3D structural mechanics (COMSOL FEMLAB) with complex geometry.
    # ~72 NNZ/row mean but high variance due to boundary/interior coupling.
    "sme3Db":      ("spgemm_ss", "FEMLAB",  29067,  2081063, 2.46e-3, "power-law",  "Structural Problem"),
    # Circuit DAE with BDF+Newton; power-law degree from hub nets (VDD/GND).
    # STD of NNZ/row is very high — few nets connect to many components.
    "circuit_2":   ("spgemm_ss", "Bomhof",   4510,    21199, 1.04e-3, "power-law",  "Circuit Simulation"),
    # MSC/NASTRAN knuckle joint — complex 3D assembly with irregular mesh.
    # Structural FEM; hub DOFs at joint interfaces drive high degree STD.
    "msc10848":    ("spgemm_ss", "Boeing",  10848,  1229776, 1.04e-2, "power-law",  "Structural Problem"),
    # DNVS topology optimization benchmark (symmetric structural stiffness matrix).
    "m_t1":        ("spgemm_ss", "DNVS",    97578,  9753570, 1.02e-3, "power-law",  "Structural Problem"),
    # Multibody dynamics structural matrix from Rothberg collection.
    "gearbox":     ("spgemm_ss", "Rothberg",153746, 9080404, 3.84e-4, "power-law",  "Structural Problem"),
    # Interior-point optimization KKT matrix (Gupta collection).
    "gupta3":      ("spgemm_ss", "Gupta",   16783,  9323427, 3.31e-2, "power-law",  "Optimization Problem"),
}

# SuiteSparse Matrix Market download URL template
_SS_MM_URL = "https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz"

# SNAP download URLs  (name -> url)
_SNAP_URLS: dict[str, str] = {
    "cit-HepTh":        "https://snap.stanford.edu/data/cit-HepTh.txt.gz",
    "cit-HepPh":        "https://snap.stanford.edu/data/cit-HepPh.txt.gz",
    "email-Enron":      "https://snap.stanford.edu/data/email-Enron.txt.gz",
    "soc-Epinions1":    "https://snap.stanford.edu/data/soc-Epinions1.txt.gz",
    "soc-Slashdot0811": "https://snap.stanford.edu/data/soc-Slashdot0811.txt.gz",
}

_GO_OBO_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _rename_root(src: Path, dst: Path) -> Path:
    """Rename a downloaded dataset folder to `dst`; skip if already done."""
    if dst.exists():
        print(f"  [skip] '{dst}' already exists.")
        return dst
    
    assert src.exists(), f"Source directory '{src}' does not exist."
    src.rename(dst)
    print(f"  Renamed '{src.name}' -> '{dst.name}'")
    return dst


def load_ogb(name: str, root: Path) -> Path:
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError:
        sys.exit(
            "ERROR: 'ogb' package not found. Install it with:  pip install ogb"
        )
    # OGB downloads into root/<name>/
    tmp_root = root / "ogb_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {name} via OGB ...")
    dataset = PygNodePropPredDataset(name=name, root=str(tmp_root))
    data = dataset[0]
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    # The actual directory OGB creates is root/ogb_tmp/<name>/
    ogb_dir = tmp_root / name.replace("-", "_")
    final_dir = root / name
    _rename_root(ogb_dir, final_dir)
    # Remove the empty tmp wrapper if empty
    try:
        tmp_root.rmdir()
    except OSError:
        pass
    print(f"  {name}: nodes={num_nodes:,}  edges={num_edges:,}")
    return final_dir


def load_pyg_reddit(name: str, root: Path) -> Path:
    from torch_geometric.datasets import Reddit
    dl_root = root / name
    dl_root.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading Reddit ...")
    dataset = Reddit(root=str(dl_root))
    data = dataset[0]
    print(f"  Reddit: nodes={data.num_nodes:,}  edges={data.num_edges:,}")
    return dl_root


def load_pyg_flickr(name: str, root: Path) -> Path:
    from torch_geometric.datasets import Flickr
    dl_root = root / name
    dl_root.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading Flickr ...")
    dataset = Flickr(root=str(dl_root))
    data = dataset[0]
    print(f"  Flickr: nodes={data.num_nodes:,}  edges={data.num_edges:,}")
    return dl_root


def load_pyg_planetoid(name: str, root: Path) -> Path:
    from torch_geometric.datasets import Planetoid
    tmp = root / "planetoid_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {name} (Planetoid) ...")
    dataset = Planetoid(root=str(tmp), name=name)
    data = dataset[0]
    # PyG creates root/planetoid_tmp/<Name>/
    src = tmp / name
    dst = root / name
    _rename_root(src, dst)
    try:
        tmp.rmdir()
    except OSError:
        pass
    print(f"  {name}: nodes={data.num_nodes:,}  edges={data.num_edges:,}")
    return dst


def load_pyg_amazon(name: str, root: Path) -> Path:
    from torch_geometric.datasets import Amazon
    # name is e.g. "Amazon-Computers" -> PyG subset = "Computers"
    subset = name.split("-", 1)[1]   # "Computers" or "Photo"
    tmp = root / "amazon_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {name} (Amazon/{subset}) ...")
    dataset = Amazon(root=str(tmp), name=subset)
    data = dataset[0]
    src = tmp / subset
    dst = root / name
    _rename_root(src, dst)
    try:
        tmp.rmdir()
    except OSError:
        pass
    print(f"  {name}: nodes={data.num_nodes:,}  edges={data.num_edges:,}")
    return dst


def load_pyg_coauthor(name: str, root: Path) -> Path:
    from torch_geometric.datasets import Coauthor
    # name is e.g. "Coauthor-CS" -> subset = "CS"
    subset = name.split("-", 1)[1]
    tmp = root / "coauthor_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {name} (Coauthor/{subset}) ...")
    dataset = Coauthor(root=str(tmp), name=subset)
    data = dataset[0]
    src = tmp / subset
    dst = root / name
    _rename_root(src, dst)
    try:
        tmp.rmdir()
    except OSError:
        pass
    print(f"  {name}: nodes={data.num_nodes:,}  edges={data.num_edges:,}")
    return dst


# ---------------------------------------------------------------------------
# Transitive Closure Loaders
# Output: <root>/<name>/  containing:
#   edges.npy     — int32 array of shape (E, 2), each row = [src, dst]
#   metadata.txt  — num_nodes, num_edges, degree_type
# ---------------------------------------------------------------------------

def _download_url(url: str, dest: Path) -> None:
    """Download a URL to dest, showing progress."""
    print(f"  Fetching {url} ...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; dataset-downloader/1.0)"},
    )
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 65536
        with open(dest, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r    {downloaded/1024/1024:.1f} / {total/1024/1024:.1f} MB  ({pct:.0f}%)", end="", flush=True)
    print()


def _parse_snap_edgelist(gz_path: Path) -> np.ndarray:
    """
    Parse a SNAP .txt.gz edge list (comment lines start with # or %).
    Returns int32 array of shape (E, 2).
    """
    edges = []
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    edges.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    continue
    return np.array(edges, dtype=np.int32)


def _remap_nodes(edges: np.ndarray) -> tuple[np.ndarray, int]:
    """Remap node IDs to contiguous 0-based integers. Returns (remapped_edges, num_nodes)."""
    unique_nodes = np.unique(edges)
    node_map = {old: new for new, old in enumerate(unique_nodes)}
    remapped = np.array([[node_map[s], node_map[d]] for s, d in edges], dtype=np.int32)
    return remapped, len(unique_nodes)


# ---------------------------------------------------------------------------
# SpGEMM Loader — SuiteSparse Matrix Market (.mtx)
# Output: <root>/<name>/
#   A.npz        — scipy sparse CSR matrix (rows, cols, data)
#   edges.npy    — int32 array shape (NNZ, 2): [row, col] indices
#   metadata.txt — num_nodes, nnz, density, degree_type, domain
# ---------------------------------------------------------------------------

def load_spgemm_ss(name: str, root: Path) -> Path:
    """Download a SuiteSparse matrix in Matrix Market format for SpGEMM experiments."""
    import tarfile

    out_dir = root / name
    edges_file = out_dir / "edges.npy"
    if edges_file.exists():
        print(f"  [skip] '{out_dir}' already exists.")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    _, group, approx_n, approx_nnz, density, degree_type, domain = SPGEMM_DATASETS[name]
    url = _SS_MM_URL.format(group=group, name=name)
    tar_path = out_dir / f"{name}.tar.gz"

    _download_url(url, tar_path)
    print(f"  Extracting archive ...")

    # Extract .tar.gz — contains <name>/<name>.mtx
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=out_dir)
    tar_path.unlink()

    # Find the extracted .mtx file
    mtx_candidates = list(out_dir.rglob("*.mtx"))
    if not mtx_candidates:
        raise FileNotFoundError(f"No .mtx file found after extracting {name}")
    mtx_path = mtx_candidates[0]

    print(f"  Parsing Matrix Market file ...")
    rows_list, cols_list = [], []
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
                # Header: M N NNZ
                num_rows, num_cols = int(parts[0]), int(parts[1])
                data_started = True
                continue
            # Edge line: row col [value]
            r, c = int(parts[0]) - 1, int(parts[1]) - 1   # 1-based → 0-based
            rows_list.append(r)
            cols_list.append(c)
            if is_symmetric and r != c:
                rows_list.append(c)
                cols_list.append(r)

    mtx_path.unlink()
    # Remove extracted subdirectory if empty
    for d in [p for p in out_dir.iterdir() if p.is_dir()]:
        try:
            d.rmdir()
        except OSError:
            pass

    edges = np.column_stack([
        np.array(rows_list, dtype=np.int32),
        np.array(cols_list, dtype=np.int32),
    ])
    nnz = len(edges)
    actual_density = nnz / (num_rows * num_cols) if num_rows > 0 else 0.0

    np.save(edges_file, edges)
    with open(out_dir / "metadata.txt", "w") as f:
        f.write(f"name={name}\n")
        f.write(f"num_rows={num_rows}\n")
        f.write(f"num_cols={num_cols}\n")
        f.write(f"nnz={nnz}\n")
        f.write(f"density={actual_density:.4e}\n")
        f.write(f"degree_type={degree_type}\n")
        f.write(f"domain={domain}\n")
        f.write(f"source=SuiteSparse Matrix Collection\n")
        f.write(f"suitesparse_group={group}\n")
        f.write(f"url={url}\n")

    print(f"  {name}: rows={num_rows:,}  nnz={nnz:,}  density={actual_density:.2e}  [{degree_type}]")
    return out_dir


def load_tc_snap(name: str, root: Path) -> Path:
    """Download a SNAP edge-list dataset for transitive closure experiments."""
    out_dir = root / name
    edges_file = out_dir / "edges.npy"
    if edges_file.exists():
        print(f"  [skip] '{out_dir}' already exists.")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    url = _SNAP_URLS[name]
    gz_path = out_dir / f"{name}.txt.gz"

    _download_url(url, gz_path)
    print(f"  Parsing edge list ...")
    edges = _parse_snap_edgelist(gz_path)
    edges, num_nodes = _remap_nodes(edges)
    num_edges = len(edges)

    np.save(edges_file, edges)
    gz_path.unlink()   # remove raw gz after parsing

    _, _, _, degree_type = TC_DATASETS[name]
    with open(out_dir / "metadata.txt", "w") as f:
        f.write(f"name={name}\n")
        f.write(f"num_nodes={num_nodes}\n")
        f.write(f"num_edges={num_edges}\n")
        f.write(f"degree_type={degree_type}\n")
        f.write(f"source=SNAP\n")
        f.write(f"url={url}\n")

    print(f"  {name}: nodes={num_nodes:,}  edges={num_edges:,}  [{degree_type}]")
    return out_dir


def load_tc_go(name: str, root: Path) -> Path:
    """
    Download Gene Ontology (go-basic.obo) and extract the 'is_a' DAG.
    Requires no external packages — parses the OBO format directly.
    """
    out_dir = root / name
    edges_file = out_dir / "edges.npy"
    if edges_file.exists():
        print(f"  [skip] '{out_dir}' already exists.")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    obo_path = out_dir / "go-basic.obo"

    _download_url(_GO_OBO_URL, obo_path)
    print(f"  Parsing OBO file ...")

    # Parse OBO: collect GO term IDs and is_a edges
    term_ids: dict[str, int] = {}   # GO:XXXXXXX -> int index
    edges_raw: list[tuple[int, int]] = []
    current_id: str | None = None
    in_term = False

    with open(obo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                in_term = True
                current_id = None
            elif line.startswith("[") and line != "[Term]":
                in_term = False
                current_id = None
            elif in_term:
                if line.startswith("id: GO:"):
                    go_id = line.split("id: ", 1)[1].strip()
                    if go_id not in term_ids:
                        term_ids[go_id] = len(term_ids)
                    current_id = go_id
                elif line.startswith("is_a:") and current_id is not None:
                    parent = line.split("is_a:", 1)[1].strip().split(" ")[0]
                    if parent not in term_ids:
                        term_ids[parent] = len(term_ids)
                    # Edge direction: child -> parent (reachability = transitive closure)
                    edges_raw.append((term_ids[current_id], term_ids[parent]))
                elif line.startswith("is_obsolete: true"):
                    current_id = None   # skip obsolete terms

    edges = np.array(edges_raw, dtype=np.int32)
    num_nodes = len(term_ids)
    num_edges = len(edges)

    np.save(edges_file, edges)
    # Save term-ID mapping
    with open(out_dir / "term_ids.txt", "w") as f:
        for go_id, idx in term_ids.items():
            f.write(f"{idx}\t{go_id}\n")
    obo_path.unlink()

    _, _, _, degree_type = TC_DATASETS[name]
    with open(out_dir / "metadata.txt", "w") as f:
        f.write(f"name={name}\n")
        f.write(f"num_nodes={num_nodes}\n")
        f.write(f"num_edges={num_edges}\n")
        f.write(f"degree_type={degree_type}\n")
        f.write(f"source=Gene Ontology (go-basic.obo)\n")
        f.write(f"url={_GO_OBO_URL}\n")

    print(f"  {name}: nodes={num_nodes:,}  edges={num_edges:,}  [{degree_type}]")
    return out_dir


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

LOADERS = {
    # GNN loaders
    "ogb":            load_ogb,
    "pyg_reddit":     load_pyg_reddit,
    "pyg_flickr":     load_pyg_flickr,
    "pyg_planetoid":  load_pyg_planetoid,
    "pyg_amazon":     load_pyg_amazon,
    "pyg_coauthor":   load_pyg_coauthor,
    # Transitive Closure loaders
    "tc_snap":        load_tc_snap,
    "tc_go":          load_tc_go,
    # SpGEMM loaders
    "spgemm_ss":      load_spgemm_ss,
}


def download_gnn(name: str, root: Path) -> None:
    if name not in GNN_DATASETS:
        print(f"ERROR: Unknown GNN dataset '{name}'. Run with --task gnn --list.")
        return
    loader_key, approx_nodes = GNN_DATASETS[name]
    loader_fn = LOADERS[loader_key]
    print(f"\n[{name}]  (~{approx_nodes:,} nodes)")
    try:
        out = loader_fn(name, root)
        print(f"  Saved to: {out}")
    except Exception as exc:
        print(f"  ERROR downloading {name}: {exc}")


def download_tc(name: str, root: Path) -> None:
    if name not in TC_DATASETS:
        print(f"ERROR: Unknown TC dataset '{name}'. Run with --task tc --list.")
        return
    loader_key, approx_nodes, approx_edges, degree_type = TC_DATASETS[name]
    loader_fn = LOADERS[loader_key]
    print(f"\n[{name}]  (~{approx_nodes:,} nodes, ~{approx_edges:,} edges, {degree_type})")
    try:
        out = loader_fn(name, root)
        print(f"  Saved to: {out}")
    except Exception as exc:
        print(f"  ERROR downloading {name}: {exc}")


def download_spgemm(name: str, root: Path) -> None:
    if name not in SPGEMM_DATASETS:
        print(f"ERROR: Unknown SpGEMM dataset '{name}'. Run with --task spgemm --list.")
        return
    _, _, approx_n, approx_nnz, density, degree_type, domain = SPGEMM_DATASETS[name]
    loader_fn = LOADERS["spgemm_ss"]
    print(f"\n[{name}]  (~{approx_n:,} nodes, ~{approx_nnz:,} NNZ, density={density:.2e}, {degree_type})")
    try:
        out = loader_fn(name, root)
        print(f"  Saved to: {out}")
    except Exception as exc:
        print(f"  ERROR downloading {name}: {exc}")


# Keep old entry point for backward compat
def download(name: str, root: Path) -> None:
    download_gnn(name, root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download graph datasets for GNN or Transitive Closure simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task",
        choices=["gnn", "tc", "spgemm"],
        default="gnn",
        help=(
            "Which task to download datasets for.\n"
            "  gnn    → saved under ./gnn  (default)\n"
            "  tc     → saved under ./tc\n"
            "  spgemm → saved under ./spgemm"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets for the selected task.",
    )
    group.add_argument(
        "--datasets",
        nargs="+",
        metavar="NAME",
        help="One or more dataset names to download (space-separated).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Override the base output directory (parent of HS/ and MS/). "
            "Defaults to the directory containing this script. "
            "HS/ and MS/ subdirectories are always created beneath it."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets for the selected task and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task: str = args.task

    # ── Determine registry and sparse-category roots ───────────────────────
    base_dir = Path(__file__).parent
    if task == "gnn":
        registry = GNN_DATASETS
    elif task == "tc":
        registry = TC_DATASETS
    else:  # spgemm
        registry = SPGEMM_DATASETS

    # HS = Highly Sparse, MS = Moderately Sparse
    # tc → HS, spgemm → MS, gnn → split per dataset
    if args.root is not None:
        hs_root = (args.root / "HS").resolve()
        ms_root = (args.root / "MS").resolve()
    else:
        hs_root = (base_dir / "HS").resolve()
        ms_root = (base_dir / "MS").resolve()

    # ── --list ────────────────────────────────────────────────────────────────
    if args.list:
        if task == "gnn":
            print(f"Available GNN datasets")
            print(f"  HS = {hs_root}")
            print(f"  MS = {ms_root}")
            print(f"  {'Name':<25} {'~Nodes':>10}  Category")
            print("  " + "-" * 48)
            for name, (_, nodes) in GNN_DATASETS.items():
                cat = "MS" if name in GNN_MS_DATASETS else "HS"
                print(f"  {name:<25} {nodes:>10,}  {cat}")
        elif task == "tc":
            print(f"Available Transitive Closure datasets  (output: {hs_root})")
            print(f"  {'Name':<22} {'~Nodes':>8} {'~Edges':>10}  Degree Type")
            print("  " + "-" * 58)
            for name, (_, nodes, edges, dtype) in TC_DATASETS.items():
                marker = "★" if dtype == "regular" else "↗"
                print(f"  {marker} {name:<20} {nodes:>8,} {edges:>10,}  {dtype}")
            print("\n  ★ = Regular (low STD)   ↗ = Power-Law (high STD)")
        else:  # spgemm
            print(f"Available SpGEMM datasets  (output: {ms_root})")
            print(f"  {'Name':<14} {'~Nodes':>8} {'~NNZ':>10} {'Density':>10}  Deg.Type    Domain")
            print("  " + "-" * 75)
            for name, (_, grp, nodes, nnz, dens, dtype, domain) in SPGEMM_DATASETS.items():
                marker = "★" if dtype == "regular" else "↗"
                print(f"  {marker} {name:<12} {nodes:>8,} {nnz:>10,} {dens:>10.2e}  {dtype:<11} {domain}")
            print("\n  ★ = Regular (low STD)   ↗ = Power-Law (high STD)")
        return

    # ── Determine targets ────────────────────────────────────────────────────
    if args.all:
        targets = list(registry.keys())
    elif args.datasets:
        targets = args.datasets
    else:
        print(
            f"No datasets specified. Use --all or --datasets NAME [...] or --list.\n"
            f"Example: python download_datasets.py --task {task} --list"
        )
        return

    print(f"Task:     {task}")
    print(f"HS root:  {hs_root}")
    print(f"MS root:  {ms_root}")

    # ── Download ──────────────────────────────────────────────────────────────
    if task == "gnn":
        for name in targets:
            root = ms_root if name in GNN_MS_DATASETS else hs_root
            root.mkdir(parents=True, exist_ok=True)
            download_gnn(name, root)
    elif task == "tc":
        hs_root.mkdir(parents=True, exist_ok=True)
        for name in targets:
            download_tc(name, hs_root)
    else:  # spgemm
        ms_root.mkdir(parents=True, exist_ok=True)
        for name in targets:
            download_spgemm(name, ms_root)

    print("\nDone.")


if __name__ == "__main__":
    main()