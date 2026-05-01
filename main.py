import collections
import os
import sys
import time
import threading

import cv2
import torch

from src.capture import get_frames
from src.display import draw_overlay
from src.landmarks import HolisticLandmarkExtractor, HOLISTIC_VEC_SIZE
from src.llm import words_to_sentence
from src.model import LandmarkTransformer, CLASS_LABELS
from src.smoothing import PredictionSmoother

SEQ_LEN = 30
CHECKPOINT_PATH = os.path.join("checkpoints", "best_model.pt")
SENTENCE_TIMEOUT = 5.0  # seconds of no new words before sentence resets


def main() -> None:
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: No trained model found at {CHECKPOINT_PATH}")
        print("Run 'python train.py' first to train the model.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LandmarkTransformer(
        num_classes=len(CLASS_LABELS),
        input_size=HOLISTIC_VEC_SIZE,
        seq_len=SEQ_LEN,
    ).to(device)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    smoother = PredictionSmoother(num_classes=len(CLASS_LABELS))
    landmark_buffer: collections.deque = collections.deque(maxlen=SEQ_LEN)

    word_buffer: list[str] = []
    last_added_word: str = ""
    current_sentence: str = ""
    last_word_time: float = time.time()
    llm_running: bool = False

    prev_time = time.time()
    fps = 0.0

    def update_sentence(words: list[str]) -> None:
        nonlocal current_sentence, llm_running
        try:
            current_sentence = words_to_sentence(words)
        except Exception:
            pass
        finally:
            llm_running = False

    with HolisticLandmarkExtractor() as extractor:
        for ok, frame in get_frames(camera_id=0, width=640, height=480):
            if not ok or frame is None:
                break

            vec, drawn = extractor.process_and_draw(frame)

            if vec is not None:
                landmark_buffer.append(vec)

            now = time.time()
            dt = now - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = now

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

                if is_confident and sign_name and sign_name != last_added_word:
                    word_buffer.append(sign_name)
                    last_added_word = sign_name
                    last_word_time = now
                    if not llm_running:
                        llm_running = True
                        t = threading.Thread(
                            target=update_sentence,
                            args=(word_buffer.copy(),),
                            daemon=True,
                        )
                        t.start()

            # Reset sentence after timeout with no new words
            if word_buffer and (now - last_word_time) > SENTENCE_TIMEOUT:
                word_buffer.clear()
                last_added_word = ""
                current_sentence = ""

            drawn = draw_overlay(
                drawn, sign_name, confidence, is_confident, fps, current_sentence
            )

            cv2.imshow("ASL Interpreter", drawn)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                word_buffer.clear()
                last_added_word = ""
                current_sentence = ""

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    sys.exit(0)
