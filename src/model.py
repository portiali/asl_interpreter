"""
Simple LSTM that takes sequences of MediaPipe landmark vectors.

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


class LandmarkLSTM(nn.Module):
    """
    LSTM over sequences of pose landmark vectors.
    Input: (batch, seq_len, LANDMARK_VEC_SIZE)
    Output: (batch, num_classes) for classification, or (batch, seq_len, hidden) for sequence output.
    """

    def __init__(
        self,
        input_size: int = HOLISTIC_VEC_SIZE,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(out_size, out_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        returns: (batch, num_classes)
        """
        out, _ = self.lstm(x)
        # use last timestep
        last = out[:, -1, :]
        return self.fc(last)


def build_dummy_sequence(batch: int = 2, seq_len: int = 30) -> torch.Tensor:
    """Create a dummy (batch, seq_len, 126) tensor for testing."""
    return torch.randn(batch, seq_len, HOLISTIC_VEC_SIZE)
