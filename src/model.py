"""
Transforemer architecture that takes sequences of MediaPipe landmark vectors.

Supports both pose landmarks (132-d) and hand landmarks (126-d).
"""

import torch
import torch.nn as nn
from src.landmarks import HOLISTIC_VEC_SIZE

CLASS_LABELS = {
    0: "hello",
    1: "thank_you",
    2: "yes",
    3: "no",
    4: "i_love_you",
}


class LandmarkTransformer(nn.Module):
    """
    Transformer over sequences of pose landmark vectors.
    Input: (batch, seq_len, LANDMARK_VEC_SIZE)
    Output: (batch, num_classes) for classification, or (batch, seq_len, hidden) for sequence output.
    """

    def __init__(
        self,
        num_classes: int,
        input_size: int = HOLISTIC_VEC_SIZE,
        d_model: int = 256,  # We can change this if it underfits
        nhead: int = 4,
        dim_feedforward: int = 1024,
        num_layers: int = 4,
        dropout: float = 0.2,
        seq_len: int = 30,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.proj = nn.Linear(input_size, self.d_model)

        self.pos_embed = nn.Embedding(seq_len, d_model)

        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.layer, num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        returns: (batch, num_classes)
        """
        # Convert HOLISTIC_VECTOR_SIZE -> d_model
        x = self.proj(x)

        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_embed(positions)

        # Self-attention encoder
        x = self.transformer(x)

        # Mean-pool across time (every frame has attended to every other)
        x = x.mean(dim=1)

        return self.fc(x)


def build_dummy_sequence(batch: int = 2, seq_len: int = 30) -> torch.Tensor:
    """Create a dummy (batch, seq_len, 126) tensor for testing."""
    return torch.randn(batch, seq_len, HOLISTIC_VEC_SIZE)
