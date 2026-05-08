"""
Data loading and split for scFoundation cell-type annotation baseline.

Input : /lichaohan/readData/5w_allcelltype_anno_symbol.h5ad
          - 50,000 cells × 30,582 genes (var['gene_symbol'] = HGNC symbol)
          - X: log1p normalized (float32)
          - obs['cell_type']: 29 classes

Pipeline:
  1. Extract X as DataFrame indexed by gene_symbol
  2. Align to scFoundation 19,264 gene order via main_gene_selection()
     (missing genes → zero-padded; extra genes → dropped)
  3. Stratified 80/20 split — identical logic to collaborator's datamodule
  4. Return DataLoaders + class metadata + class weights (computed but not used
     in default training; standard CrossEntropyLoss is used instead)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import issparse

# ── Inlined from scFoundation/model/get_embedding.py ─────────────────────
# (Cannot import directly: get_embedding.py calls argparse.parse_args() at
#  module level, which would intercept our own CLI arguments.)
def main_gene_selection(X_df, gene_list):
    """Reorder/zero-pad X_df columns to exactly match gene_list order."""
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(
        np.zeros((X_df.shape[0], len(to_fill_columns))),
        columns=to_fill_columns,
        index=X_df.index,
    )
    X_df = pd.DataFrame(
        np.concatenate([X_df.values, padding_df.values], axis=1),
        index=X_df.index,
        columns=list(X_df.columns) + list(padding_df.columns),
    )
    X_df = X_df[gene_list]
    var = pd.DataFrame(index=X_df.columns)
    var["mask"] = [1 if g in to_fill_columns else 0 for g in var.index]
    return X_df, to_fill_columns, var

# ── Paths ──────────────────────────────────────────────────────────────────
H5AD_PATH       = "/lichaohan/readData/5w_allcelltype_anno_symbol.h5ad"
GENE_INDEX_PATH = "/lichaohan/scFoundation/OS_scRNA_gene_index.19264.tsv"


# ── Exact replica of collaborator's _stratified_train_val_split ────────────
def _stratified_train_val_split(
    barcodes: np.ndarray,
    labels: np.ndarray,
    train_size: float = 0.8,
    random_state: int = 42,
):
    """
    Deterministic stratified split. Ensures ≥1 sample per class per split.
    Returns (train_barcodes, val_barcodes) as np.ndarray of barcode strings.
    """
    barcodes = np.asarray(barcodes)
    labels   = np.asarray(labels)

    rng = np.random.default_rng(int(random_state))
    train_parts, val_parts = [], []

    for lab in np.unique(labels):
        idx  = np.flatnonzero(labels == lab)
        perm = rng.permutation(idx)
        n_tr = int(np.floor(len(idx) * train_size))
        if len(idx) >= 2:
            n_tr = min(max(n_tr, 1), len(idx) - 1)
        else:
            n_tr = len(idx)
        train_parts.append(perm[:n_tr])
        val_parts.append(perm[n_tr:])

    train_idx = np.concatenate(train_parts)
    val_idx   = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return barcodes[train_idx], barcodes[val_idx]


def _stratified_train_val_test_split(
    barcodes: np.ndarray,
    labels: np.ndarray,
    train_size: float = 0.8,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """
    Deterministic stratified train/val/test split.

    The default 0.8/0.1/0.1 split matches the evaluation protocol used by
    read_nt_v3 while keeping at least one held-out sample per class when the
    class has enough cells.
    """
    if not 0.0 < train_size < 1.0:
        raise ValueError(f"train_size must be in (0, 1), got {train_size}")
    if not 0.0 <= val_size < 1.0:
        raise ValueError(f"val_size must be in [0, 1), got {val_size}")
    if train_size + val_size >= 1.0:
        raise ValueError("train_size + val_size must be < 1.0")

    barcodes = np.asarray(barcodes)
    labels = np.asarray(labels)
    rng = np.random.default_rng(int(random_state))
    train_parts, val_parts, test_parts = [], [], []

    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        perm = rng.permutation(idx)
        n = len(perm)

        if n < 3:
            n_tr = 1 if n >= 2 else n
            n_val = n - n_tr
        else:
            n_tr = int(np.floor(n * train_size))
            n_val = int(np.floor(n * val_size))
            n_tr = min(max(n_tr, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_tr - 1)

        train_parts.append(perm[:n_tr])
        val_parts.append(perm[n_tr:n_tr + n_val])
        test_parts.append(perm[n_tr + n_val:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return barcodes[train_idx], barcodes[val_idx], barcodes[test_idx]


# ── PyTorch Dataset ────────────────────────────────────────────────────────
class CellTypeDataset(Dataset):
    """Single-cell gene expression dataset for scFoundation classifier."""

    def __init__(self, X: np.ndarray, labels: np.ndarray):
        self.X      = torch.tensor(X,      dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 'x' key matches scFoundation's forward() signature
        return {"x": self.X[idx], "targets": self.labels[idx]}


# ── Main loader ────────────────────────────────────────────────────────────
def load_data(
    h5ad_path:       str   = H5AD_PATH,
    gene_index_path: str   = GENE_INDEX_PATH,
    train_size:      float = 0.8,
    val_size:        float = 0.1,
    random_state:    int   = 42,
    batch_size:      int   = 32,
    num_workers:     int   = 4,
    return_test:     bool  = False,
):
    """
    Returns
    -------
    train_loader, val_loader : DataLoader
    test_loader              : DataLoader (only when return_test=True)
    class_names              : list[str]  – sorted cell-type names
    type2idx                 : dict[str, int]
    class_weights            : np.ndarray (n_class,)  – inverse-frequency weights
                               (returned for reference; standard CE is used in training)
    """
    import anndata as ad

    print(f"Loading h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}  obs columns: {list(adata.obs.columns)}")

    # ── 1. Build gene-symbol-indexed DataFrame ────────────────────────────
    X_raw = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
    # var index is Ensembl; use gene_symbol as column name for alignment
    X_df  = pd.DataFrame(X_raw,
                         index=adata.obs.index,
                         columns=adata.var["gene_symbol"].values)

    # ── 2. Align to scFoundation 19,264 gene order ────────────────────────
    gene_list_df = pd.read_csv(gene_index_path, sep="\t")
    gene_list    = list(gene_list_df["gene_name"])
    X_df, missing_genes, _ = main_gene_selection(X_df, gene_list)
    print(f"  Gene alignment: {len(gene_list)} genes total, "
          f"{len(missing_genes)} zero-padded (not in our data)")
    assert X_df.shape[1] == 19264, f"Expected 19264 columns, got {X_df.shape[1]}"

    # ── 3. Encode labels ──────────────────────────────────────────────────
    cell_types  = adata.obs["cell_type"].values
    class_names = sorted(set(cell_types))
    type2idx    = {t: i for i, t in enumerate(class_names)}
    labels      = np.array([type2idx[t] for t in cell_types])
    barcodes    = np.array(adata.obs.index.tolist())

    # ── 4. Stratified split ───────────────────────────────────────────────
    if return_test:
        train_bc, val_bc, test_bc = _stratified_train_val_test_split(
            barcodes,
            labels,
            train_size=train_size,
            val_size=val_size,
            random_state=random_state,
        )
    else:
        train_bc, val_bc = _stratified_train_val_split(
            barcodes, labels, train_size=train_size, random_state=random_state
        )
        test_bc = np.array([], dtype=barcodes.dtype)

    train_set = set(train_bc.tolist())
    val_set = set(val_bc.tolist())
    test_set = set(test_bc.tolist())
    train_mask = np.array([b in train_set for b in barcodes])
    val_mask = np.array([b in val_set for b in barcodes])
    test_mask = np.array([b in test_set for b in barcodes])
    if not return_test:
        val_mask = ~train_mask

    X_arr = X_df.values.astype(np.float32)
    train_ds = CellTypeDataset(X_arr[train_mask], labels[train_mask])
    val_ds   = CellTypeDataset(X_arr[val_mask],   labels[val_mask])
    test_ds  = CellTypeDataset(X_arr[test_mask],  labels[test_mask])

    if return_test:
        print(f"  Split: {train_mask.sum()} train / {val_mask.sum()} val / "
              f"{test_mask.sum()} test")
    else:
        print(f"  Split: {train_mask.sum()} train / {val_mask.sum()} val")

    # ── 5. Class weights (inverse frequency, returned for reference) ─────────
    train_labels  = labels[train_mask]
    n_class       = len(class_names)
    class_counts  = np.bincount(train_labels, minlength=n_class).astype(float)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()  # normalize around 1.0
    print(f"  Classes: {n_class}  |  min count={int(class_counts.min())}  "
          f"max count={int(class_counts.max())}  "
          f"max/min weight ratio={class_weights.max()/class_weights.min():.0f}x")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    if return_test:
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        return (train_loader, val_loader, test_loader,
                class_names, type2idx, class_weights)

    return train_loader, val_loader, class_names, type2idx, class_weights


if __name__ == "__main__":
    train_loader, val_loader, class_names, type2idx, class_weights = load_data()
    batch = next(iter(train_loader))
    print(f"Batch x shape: {batch['x'].shape}")
    print(f"Batch labels:  {batch['targets'][:8]}")
    print(f"Class names ({len(class_names)}): {class_names[:5]} ...")
