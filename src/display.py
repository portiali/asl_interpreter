"""
Polished display overlay for the ASL interpreter demo.
"""

import cv2
import numpy as np


def draw_overlay(
    frame: np.ndarray,
    sign_name: str,
    confidence: float,
    is_confident: bool,
    fps: float,
) -> np.ndarray:
    """Draw demo overlay on a frame.

    Args:
        frame: BGR image to draw on (modified in place).
        sign_name: Name of the predicted sign.
        confidence: Confidence value between 0 and 1.
        is_confident: Whether the prediction meets the confidence threshold.
        fps: Current frames per second.

    Returns:
        The frame with overlay drawn.
    """
    h, w = frame.shape[:2]

    # --- Bottom bar (semi-transparent dark background) ---
    bar_height = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if is_confident:
        # Sign name
        display_name = sign_name.replace("_", " ").upper()
        cv2.putText(
            frame,
            display_name,
            (20, h - 45),
            cv2.FONT_HERSHEY_DUPLEX,
            1.4,
            (255, 255, 255),
            2,
        )

        # Confidence bar background
        bar_x = 20
        bar_y = h - 30
        bar_w = 200
        bar_h = 16
        cv2.rectangle(
            frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1
        )

        # Confidence bar fill (green > 80%, yellow > 60%)
        fill_w = int(bar_w * confidence)
        if confidence > 0.8:
            color = (0, 200, 0)
        elif confidence > 0.6:
            color = (0, 200, 200)
        else:
            color = (0, 0, 200)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

        # Confidence percentage
        pct_text = f"{confidence * 100:.0f}%"
        cv2.putText(
            frame,
            pct_text,
            (bar_x + bar_w + 10, bar_y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    else:
        cv2.putText(
            frame,
            "...",
            (20, h - 30),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            (150, 150, 150),
            2,
        )

    # --- Top bar ---
    # Title
    cv2.putText(
        frame,
        "ASL Interpreter",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # FPS counter
    fps_text = f"{fps:.0f} FPS"
    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(
        frame,
        fps_text,
        (w - text_size[0] - 10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    return frame
