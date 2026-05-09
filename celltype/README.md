# Cell-Type Evaluation

This directory now has two evaluation paths:

1. `train.py`: original supervised scFoundation fine-tuning. It trains the MLP
   classification head with cross-entropy and selects the best checkpoint by
   validation macro-F1.
2. `probe.py`: unified embedding-probe evaluation. It extracts frozen
   scFoundation validation cell embeddings and runs the same downstream
   `LinearSVC` cross-validation protocol used by read_nt_v3.

## Unified Probe

Default split is the original stratified `80% train / 20% val` split with
`random_state=42`. The probe is run only on validation embeddings, matching
`LatentSVCAccuracyCallback(mode=val)` in read_nt_v3:

```text
validation embeddings -> 5-fold StratifiedKFold
each fold: StandardScaler -> PCA(100) -> LinearSVC(dual=False, max_iter=2000)
```

If a fold has more than 5000 training samples, the SVC fit set is subsampled
to 5000 samples, matching the read_nt_v3 config.

Run:

```bash
python celltype/probe.py \
  --ckpt /lichaohan/scFoundation/model/models/models.ckpt \
  --h5ad /lichaohan/readData/5w_allcelltype_anno_symbol.h5ad \
  --gene_index /lichaohan/scFoundation/OS_scRNA_gene_index.19264.tsv \
  --batch_size 12 \
  --seed 42
```

Outputs are written under `celltype/outputs_probe/<run_name>/`:

- `probe_metrics.json`: mean CV train/test accuracy, balanced accuracy,
  macro-F1, weighted-F1, per-fold metrics, class filtering details, class names,
  and run arguments.
- `probe_fold_metrics.csv`: per-fold train/test accuracy and macro-F1.
- `class_names.json`: label-id mapping.

Use `--save_embeddings` to also save validation `.npy` embedding and label
arrays.
