"""
Transformer architecture over MediaPipe holistic landmark sequences.

Input: (batch, seq_len, HOLISTIC_VEC_SIZE)
Output: (batch, num_classes)
"""

import json
import os
from typing import Optional

import torch
import torch.nn as nn

from src.landmarks import HOLISTIC_VEC_SIZE


class LandmarkTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int = HOLISTIC_VEC_SIZE,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        num_layers: int = 4,
        dropout: float = 0.2,
        seq_len: int = 120,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.proj = nn.Linear(input_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_embed(positions)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


def infer_config_from_state_dict(state_dict: dict) -> tuple[int, int]:
    """Read (num_classes, seq_len) directly from the checkpoint tensors."""
    num_classes = state_dict["fc.3.weight"].shape[0]
    seq_len = state_dict["pos_embed.weight"].shape[0]
    return num_classes, seq_len


def load_label_map(search_paths: list[str]) -> tuple[dict[int, str], Optional[str]]:
    """Find label_map.json in the given paths.

    Returns (idx_to_label, source_path). If nothing is found, returns ({}, None)
    and the caller should fall back to placeholder names.
    """
    for path in search_paths:
        if os.path.isfile(path):
            with open(path) as f:
                raw = json.load(f)
            # Accept both {"word": idx} and {"idx": "word"}
            if all(isinstance(v, int) for v in raw.values()):
                return {int(v): k for k, v in raw.items()}, path
            return {int(k): v for k, v in raw.items()}, path
    return {}, None


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[LandmarkTransformer, int, int]:
    """Load checkpoint, infer architecture, return (model, num_classes, seq_len)."""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    num_classes, seq_len = infer_config_from_state_dict(state_dict)

    model = LandmarkTransformer(
        num_classes=num_classes,
        input_size=HOLISTIC_VEC_SIZE,
        seq_len=seq_len,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, num_classes, seq_len
