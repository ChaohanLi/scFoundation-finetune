"""
Cell-type classifier built on top of scFoundation's pretrained encoder.

Architecture
------------
  scFoundation encoder  (frozen, except transformer_encoder[-2])
    → max-pool over gene tokens
    → BatchNorm1d(hidden_dim, affine=False)
    → Linear(hidden_dim → 256) → ReLU → Linear(256 → n_class)

Exactly mirrors LinearProbingClassifier in ../model/finetune_model.py,
with configurable n_class (default 29 for our PBMC dataset).
"""

import os
import sys

import torch
from torch import nn

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
sys.path.insert(0, _MODEL_DIR)
from load import load_model_frommmf, gatherData  # noqa: E402


class CellTypeClassifier(nn.Module):
    """
    scFoundation encoder + classification head for cell-type annotation.

    Parameters
    ----------
    ckpt_path  : path to models.ckpt
    n_class    : number of cell types (29 for our PBMC dataset)
    frozenmore : if True, also freeze token_emb and pos_emb (default True)
                 set False to allow full fine-tuning
    """

    def __init__(self, ckpt_path: str, n_class: int = 29, frozenmore: bool = True):
        super().__init__()
        self.ckpt_path  = ckpt_path
        self.n_class    = n_class
        self.frozenmore = frozenmore
        self._built     = False

    def build(self):
        # load.py calls model.cuda() unconditionally; neutralize it here
        # so we can place the model on the correct device ourselves.
        import torch.nn as _nn
        _orig_cuda = _nn.Module.cuda
        _nn.Module.cuda = lambda self, device=None: self  # no-op
        try:
            model, model_config = load_model_frommmf(self.ckpt_path)
        finally:
            _nn.Module.cuda = _orig_cuda  # restore
        self.model_config = model_config

        # Copy pretrained components
        self.token_emb = model.token_emb
        self.pos_emb   = model.pos_emb
        self.encoder   = model.encoder

        # Freeze strategy: all encoder layers except transformer_encoder[-2]
        # Mirrors finetune_model.py exactly — only [-2] is unfrozen, NOT [-1]
        for _, p in self.encoder.named_parameters():
            p.requires_grad = False
        for _, p in self.encoder.transformer_encoder[-2].named_parameters():
            p.requires_grad = True

        if self.frozenmore:
            for _, p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _, p in self.pos_emb.named_parameters():
                p.requires_grad = False

        hidden_dim = model_config["encoder"]["hidden_dim"]
        # ── Classification head ───────────────────────────────────────────
        # Source: scFoundation/model/finetune_model.py
        #         class LinearProbingClassifier, method build()
        #         Lines: self.norm = BatchNorm1d(...) + self.head = Sequential(...)
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False, eps=1e-6)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_class),
        )

        # Print trainable parameter count after all modules are attached.
        total  = sum(p.numel() for p in self.parameters())
        train_ = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameters: {total:,} total | {train_:,} trainable "
              f"({100*train_/total:.1f}%)")
        self._built = True

    def encode(self, sample_list: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        sample_list : dict with key 'x' of shape (B, 19264)

        Returns
        -------
        embeddings : (B, hidden_dim)
        """
        assert self._built, "Call model.build() before forward()"
        x = sample_list["x"]            # (B, 19264)

        value_labels = x > 0            # Boolean mask of expressed genes
        x, x_padding = gatherData(x, value_labels,
                                   self.model_config["pad_token_id"])

        # Positional gene IDs
        data_gene_ids    = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                           self.model_config["pad_token_id"])

        # Embeddings
        x  = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        x += self.pos_emb(position_gene_ids)

        # Encoder
        x = self.encoder(x, x_padding)   # (B, seq_len, hidden_dim)

        # Aggregate: max-pool over gene tokens
        x, _ = torch.max(x, dim=1)       # (B, hidden_dim)
        x = self.norm(x)
        return x

    def forward(self, sample_list: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        sample_list : dict with key 'x' of shape (B, 19264)

        Returns
        -------
        logits : (B, n_class)
        """
        x = self.encode(sample_list)
        logits = self.head(x)             # (B, n_class)
        return logits


def build_model(ckpt_path: str, n_class: int = 29,
                frozenmore: bool = True, device: str = "cuda") -> CellTypeClassifier:
    model = CellTypeClassifier(ckpt_path, n_class=n_class, frozenmore=frozenmore)
    model.build()
    model = model.to(device)
    return model
