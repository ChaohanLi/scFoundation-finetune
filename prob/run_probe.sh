#!/usr/bin/env bash
# =============================================================================
# scFoundation Probe — run script
# Edit the variables below, then execute:
#   bash run_probe.sh
# or to run in background:
#   nohup bash run_probe.sh > run_probe.log 2>&1 &
# =============================================================================
set -euo pipefail

# ─── Dataset configuration ──────────────────────────────────────────────────
#  Pick one dataset block and comment out the rest, or override all variables.

# Dataset: 5w_symbol (original, log1p normalized, 29 classes)
# H5AD="/lichaohan/readData/5w_allcelltype_anno_symbol.h5ad"
# DATASET_ID="5w_symbol"
# N_CLASS=29
# PREPROCESS=""                          # empty = no preprocessing

# Dataset: 5w_GSE196830 (raw counts, 29 classes)
# H5AD="/lichaohan/readData/5w_PBMC_GSE196830/5w_allcelltype.h5ad"
# DATASET_ID="5w_GSE196830"
# N_CLASS=29
# PREPROCESS="--preprocess"

# Dataset: GSE96583 (gene-symbol var_names, raw counts, 8 classes)
# H5AD="/lichaohan/readData/GSE96583_PBMC/GSE96583_merged_dedup.h5ad"
# DATASET_ID="GSE96583"
# N_CLASS=8
# PREPROCESS="--preprocess"

# Dataset: 10w_GSE196830 (raw counts, 29 classes)
# H5AD="/lichaohan/readData/10w_PBMC_GSE196830/10w_allcelltype.h5ad"
# DATASET_ID="10w_GSE196830"
# N_CLASS=29
# PREPROCESS="--preprocess"

# Dataset: 20w_GSE196830 (raw counts, 29 classes)
# H5AD="/lichaohan/readData/20w_PBMC_GSE196830/20w_allcelltype.h5ad"
# DATASET_ID="20w_GSE196830"
# N_CLASS=29
# PREPROCESS="--preprocess"

# Dataset: 40w_GSE196830 (raw counts, 29 classes)
H5AD="/lichaohan/readData/40w_PBMC_GSE196830/GSE196830_40w_subset.h5ad"
DATASET_ID="40w_GSE196830"
N_CLASS=29
PREPROCESS="--preprocess"

# ─── Run configuration ──────────────────────────────────────────────────────
RUN_NAME="probe"                       # wandb run name prefix (dataset_id appended automatically)
WANDB_PROJECT="scfoundation-probe"
CKPT="/lichaohan/scFoundation/model/models/models.ckpt"
GENE_INDEX="/lichaohan/scFoundation/OS_scRNA_gene_index.19264.tsv"
SYMBOL_MAP="/lichaohan/readData/gene_id_to_symbol.tsv"  # Ensembl→HGNC map for raw-count datasets
BATCH_SIZE=4                          # A100-80GB: 32 is safe; original 4 was for 16-24GB GPUs
N_JOBS=16                              # parallel CPU cores (fold parallelism)
PCA_DIM=100
MAX_ITER=2000
SAVE_EMBEDDINGS=""                     # set to "--save_embeddings" to also save embeddings_val.npy / labels_val.npy (needed for visualize.py)

PYTHON="/lichaohan/miniconda3/envs/scvi/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs_probe"   # output root; run_name appended automatically

# ─── Run ────────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"

$PYTHON probe.py \
    --h5ad        "${H5AD}" \
    --dataset_id  "${DATASET_ID}" \
    --n_class     "${N_CLASS}" \
    --ckpt        "${CKPT}" \
    --gene_index  "${GENE_INDEX}" \
    --symbol_map  "${SYMBOL_MAP}" \
    --output_dir  "${OUTPUT_DIR}" \
    --run_name    "${RUN_NAME}" \
    --wandb_project "${WANDB_PROJECT}" \
    --batch_size  "${BATCH_SIZE}" \
    --n_jobs      "${N_JOBS}" \
    --pca_dim     "${PCA_DIM}" \
    --max_iter    "${MAX_ITER}" \
    ${PREPROCESS} \
    ${SAVE_EMBEDDINGS}
