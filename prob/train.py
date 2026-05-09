"""
scFoundation cell-type annotation baseline — training script.

Usage
-----
# Linear probing (encoder frozen except transformer_encoder[-2])
python train.py --ckpt /lichaohan/scFoundation/model/models/models.ckpt

# Unfreeze embeddings too (typically not needed)
python train.py --ckpt /lichaohan/scFoundation/model/models/models.ckpt \\
    --no_frozenmore

Outputs
-------
outputs/<run_name>/
    best_model.pt        – checkpoint with best val macro-F1
    metrics.csv          – per-epoch train/val loss, macro-F1, and accuracy
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score

_CELLTYPE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CELLTYPE_DIR)
from dataset import load_data  # noqa: E402
from model   import build_model  # noqa: E402


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="scFoundation cell-type annotation")
    p.add_argument("--ckpt",       type=str,
                   default="/lichaohan/scFoundation/model/models/models.ckpt")
    p.add_argument("--h5ad",       type=str,
                   default="/lichaohan/readData/5w_allcelltype_anno_symbol.h5ad")
    p.add_argument("--gene_index", type=str,
                   default="/lichaohan/scFoundation/OS_scRNA_gene_index.19264.tsv")
    p.add_argument("--n_class",    type=int, default=29)
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--train_size", type=float, default=0.8)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--no_frozenmore", action="store_true",
                   help="Also unfreeze token/pos embeddings")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--run_name",   type=str, default=None)
    return p.parse_args()


# ── Training / evaluation helpers ─────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            x       = batch["x"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            logits = model({"x": x})
            loss   = criterion(logits, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(targets.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss   = total_loss / len(all_labels)
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy   = accuracy_score(all_labels, all_preds)
    return avg_loss, macro_f1, accuracy


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Output dir ────────────────────────────────────────────────────────
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir  = os.path.join(_CELLTYPE_DIR, args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, class_names, type2idx, _ = load_data(
        h5ad_path       = args.h5ad,
        gene_index_path = args.gene_index,
        train_size      = args.train_size,
        random_state    = args.seed,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
    )
    n_class = len(class_names)
    print(f"Effective n_class: {n_class}  (arg was {args.n_class})")
    assert n_class == args.n_class, (
        f"Expected {args.n_class} classes but found {n_class} in data. "
        "Update --n_class."
    )

    # Save class mapping
    import json
    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(
        ckpt_path  = args.ckpt,
        n_class    = n_class,
        frozenmore = not args.no_frozenmore,
        device     = device,
    )

    # ── Loss: standard cross-entropy ─────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer + LR scheduler ──────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_f1   = -1.0
    metrics_path  = os.path.join(out_dir, "metrics.csv")

    # CSV header
    with open(metrics_path, "w") as f:
        f.write("epoch,train_loss,train_macro_f1,train_acc,val_loss,val_macro_f1,val_acc\n")

    print("\n" + "=" * 60)
    print(f"Training for {args.epochs} epochs  lr={args.lr}  bs={args.batch_size}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_f1, tr_acc = run_epoch(model, train_loader, criterion,
                                             optimizer, device, training=True)
        torch.cuda.empty_cache()  # release fragmented activations before val
        val_loss, val_f1, val_acc = run_epoch(model, val_loader, criterion,
                                               optimizer, device, training=False)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss={tr_loss:.4f} f1={tr_f1:.4f} acc={tr_acc:.4f} | "
              f"val   loss={val_loss:.4f} f1={val_f1:.4f} acc={val_acc:.4f} | "
              f"{elapsed:.1f}s")

        # Save metrics
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{tr_loss:.6f},{tr_f1:.6f},{tr_acc:.6f},"
                    f"{val_loss:.6f},{val_f1:.6f},{val_acc:.6f}\n")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_path   = os.path.join(out_dir, "best_model.pt")
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc":      val_acc,
                "class_names":  class_names,
                "type2idx":     type2idx,
                "args":         vars(args),
            }, ckpt_path)
            print(f"  ↑ New best val macro-F1={val_f1:.4f} acc={val_acc:.4f}  saved to {ckpt_path}")

    print("\n" + "=" * 60)
    print(f"Training complete. Best val macro-F1: {best_val_f1:.4f}  (acc reported per epoch)")
    print(f"Metrics saved to:  {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
