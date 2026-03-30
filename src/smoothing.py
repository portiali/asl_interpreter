"""
Temporal smoothing for prediction stability.

Uses exponential moving average, confidence thresholding, and hysteresis
to prevent flickering between classes during live inference.
"""

import torch
import torch.nn.functional as F
import numpy as np


class PredictionSmoother:
    """Stabilize model predictions over time.

    Combines three mechanisms:
    - EMA over softmax outputs for smooth probability estimates
    - Confidence threshold to suppress low-confidence predictions
    - Hysteresis to prevent rapid toggling between classes
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.3,
        threshold: float = 0.6,
        hysteresis: float = 0.15,
    ):
        self.num_classes = num_classes
        self.alpha = alpha
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.ema: np.ndarray = np.ones(num_classes, dtype=np.float32) / num_classes
        self.current_class: int | None = None

    def update(self, logits: torch.Tensor) -> tuple[int, float, bool]:
        """Update smoother with new model output.

        Args:
            logits: Raw model output of shape (num_classes,) or (1, num_classes).

        Returns:
            (predicted_class, confidence, is_confident)
        """
        if logits.dim() == 2:
            logits = logits.squeeze(0)

        probs = F.softmax(logits, dim=0).detach().cpu().numpy()

        # Exponential moving average
        self.ema = self.alpha * probs + (1 - self.alpha) * self.ema

        predicted = int(np.argmax(self.ema))
        confidence = float(self.ema[predicted])

        # Hysteresis: require margin to switch away from current class
        if self.current_class is not None and predicted != self.current_class:
            current_prob = float(self.ema[self.current_class])
            if confidence - current_prob < self.hysteresis:
                predicted = self.current_class
                confidence = current_prob

        is_confident = confidence >= self.threshold

        if is_confident:
            self.current_class = predicted
        else:
            self.current_class = None

        return predicted, confidence, is_confident

    def reset(self) -> None:
        """Reset smoother state."""
        self.ema = np.ones(self.num_classes, dtype=np.float32) / self.num_classes
        self.current_class = None
