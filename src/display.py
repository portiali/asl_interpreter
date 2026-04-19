import cv2
import numpy as np


def draw_overlay(
    frame: np.ndarray,
    sign_name: str,
    confidence: float,
    is_confident: bool,
    fps: float,
    sentence: str = "",
) -> np.ndarray:
    h, w = frame.shape[:2]

    # --- Bottom bar ---
    bar_height = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if is_confident:
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

        bar_x, bar_y, bar_w, bar_h = 20, h - 30, 200, 16
        cv2.rectangle(
            frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1
        )

        fill_w = int(bar_w * confidence)
        color = (
            (0, 200, 0)
            if confidence > 0.8
            else (0, 200, 200) if confidence > 0.6 else (0, 0, 200)
        )
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

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

    # --- Sentence display ---
    if sentence:
        # Truncate if too long for screen
        max_chars = w // 11
        display_sentence = (
            sentence if len(sentence) <= max_chars else "..." + sentence[-max_chars:]
        )
        cv2.putText(
            frame,
            display_sentence,
            (20, h - 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 255, 200),
            2,
        )

    # --- Top bar ---
    cv2.putText(
        frame,
        "ASL Interpreter",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

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

    # --- Controls hint ---
    cv2.putText(
        frame,
        "Q: quit  |  R: reset",
        (10, h - 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )

    return frame
