"""
Live camera -> MediaPipe landmarks -> LSTM pipeline (starter).

Run: python main.py

- Frames are read from the default camera.
- MediaPipe Pose extracts 33 landmarks per frame (132-d vector).
- Landmarks are buffered into a sliding window and optionally fed to the LSTM.
- Press 'q' to quit.
"""
import collections
import sys

import cv2
import torch

from src.capture import get_frames
from src.landmarks import LandmarkExtractor, LANDMARK_VEC_SIZE
from src.model import LandmarkLSTM


# Sequence length the LSTM expects (e.g. 16 frames ≈ 0.5 s at 30 fps)
SEQ_LEN = 16


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandmarkLSTM(
        input_size=LANDMARK_VEC_SIZE,
        hidden_size=128,
        num_layers=2,
        num_classes=5,
        dropout=0.2,
    ).to(device)
    model.eval()

    # Sliding window of landmark vectors
    landmark_buffer: collections.deque = collections.deque(maxlen=SEQ_LEN)

    with LandmarkExtractor() as extractor:
        for ok, frame in get_frames(camera_id=0, width=640, height=480):
            if not ok or frame is None:
                break

            vec, drawn = extractor.process_and_draw(frame)

            if vec is not None:
                landmark_buffer.append(vec)
            else:
                # Optional: append zeros or skip; here we skip adding
                pass

            # Run model when we have a full sequence
            if len(landmark_buffer) == SEQ_LEN:
                seq = torch.tensor(
                    [list(landmark_buffer)],
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    logits = model(seq)
                    pred = logits.argmax(dim=1).item()
                cv2.putText(
                    drawn,
                    f"LSTM class: {pred}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Camera + Landmarks", drawn)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    sys.exit(0)
