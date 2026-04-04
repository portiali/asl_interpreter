"""
Live ASL interpreter — camera -> holistic landmarks -> LSTM -> smoothed predictions.

Run: python main.py

- Frames are read from the default camera.
- MediaPipe Holistic extracts hand + upper-body landmarks (147-d vector).
- Landmarks are buffered into a sliding window and fed to the LSTM.
- Temporal smoothing stabilizes predictions.
- Press 'q' to quit.
"""

import collections
import os
import sys
import time

import cv2
import torch

from src.capture import get_frames
from src.display import draw_overlay
from src.landmarks import HolisticLandmarkExtractor, HOLISTIC_VEC_SIZE
from src.model import LandmarkLSTM, CLASS_LABELS
from src.smoothing import PredictionSmoother

# Sequence length matching training data (~1s at 30fps)
SEQ_LEN = 30
CHECKPOINT_PATH = os.path.join("checkpoints", "best_model.pt")


def main() -> None:
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: No trained model found at {CHECKPOINT_PATH}")
        print("Run 'python train.py' first to train the model.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LandmarkLSTM(
        input_size=HOLISTIC_VEC_SIZE,
        hidden_size=128,
        num_layers=2,
        num_classes=len(CLASS_LABELS),
        dropout=0.2,
    ).to(device)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    smoother = PredictionSmoother(num_classes=len(CLASS_LABELS))

    # Sliding window of landmark vectors
    landmark_buffer: collections.deque = collections.deque(maxlen=SEQ_LEN)

    prev_time = time.time()
    fps = 0.0

    with HolisticLandmarkExtractor() as extractor:
        for ok, frame in get_frames(camera_id=0, width=640, height=480):
            if not ok or frame is None:
                break

            vec, drawn = extractor.process_and_draw(frame)

            if vec is not None:
                landmark_buffer.append(vec)

            # Track FPS
            now = time.time()
            dt = now - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = now

            # Run inference when buffer is full
            sign_name = ""
            confidence = 0.0
            is_confident = False

            if len(landmark_buffer) == SEQ_LEN:
                seq = torch.tensor(
                    [list(landmark_buffer)],
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    logits = model(seq)
                pred, confidence, is_confident = smoother.update(logits.squeeze(0))
                sign_name = CLASS_LABELS.get(pred, "unknown")

            drawn = draw_overlay(drawn, sign_name, confidence, is_confident, fps)

            cv2.imshow("ASL Interpreter", drawn)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    sys.exit(0)
