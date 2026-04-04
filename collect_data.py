"""
Record labeled holistic landmark sequences for training.

Usage:
    python collect_data.py --label hello --num_sequences 20
    python collect_data.py --label thank_you --seq_len 30 --num_sequences 20
"""

import argparse
import os
import time

import cv2
import numpy as np

from src.capture import get_frames
from src.landmarks import HolisticLandmarkExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect hand landmark sequences")
    parser.add_argument(
        "--label", type=str, required=True, help="Sign name (e.g. hello)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data", help="Base output directory"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=30,
        help="Frames per sequence (~1s at 30fps)",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=20,
        help="Number of sequences to record",
    )
    args = parser.parse_args()

    save_dir = os.path.join(args.output_dir, args.label)
    os.makedirs(save_dir, exist_ok=True)

    collected = 0
    recording = False
    buffer: list[np.ndarray] = []
    last_valid_vec: np.ndarray | None = None

    with HolisticLandmarkExtractor() as extractor:
        for ok, frame in get_frames():
            if not ok or frame is None:
                break

            vec, drawn = extractor.process_and_draw(frame)

            if vec is not None:
                last_valid_vec = vec

            if recording:
                # Use current vec, or carry forward the last valid one
                if vec is not None:
                    buffer.append(vec)
                elif last_valid_vec is not None:
                    buffer.append(last_valid_vec)
                else:
                    # No valid detection yet, abort this recording
                    recording = False
                    buffer.clear()
                    cv2.putText(
                        drawn,
                        "No hands detected - try again",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow("Collect Data", drawn)
                    cv2.waitKey(1)
                    continue

                # Show recording progress
                progress = f"Recording: {len(buffer)}/{args.seq_len}"
                cv2.putText(
                    drawn,
                    progress,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                # Sequence complete -- save it
                if len(buffer) >= args.seq_len:
                    recording = False
                    sequence = np.array(buffer, dtype=np.float32)
                    filename = f"sequence_{int(time.time() * 1000)}.npy"
                    filepath = os.path.join(save_dir, filename)
                    np.save(filepath, sequence)
                    collected += 1
                    buffer.clear()
                    print(
                        f"  Saved {filepath}  shape={sequence.shape}"
                        f"  ({collected}/{args.num_sequences})"
                    )

                    if collected >= args.num_sequences:
                        print(
                            f"\nDone! Collected {collected} sequences"
                            f" for '{args.label}' in {save_dir}/"
                        )
                        break
            else:
                # Waiting state -- show instructions
                remaining = args.num_sequences - collected
                status = (
                    f"SPACE to record '{args.label}' ({remaining} left)  |  Q to quit"
                )
                cv2.putText(
                    drawn,
                    status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                if vec is not None:
                    cv2.putText(
                        drawn,
                        "Hands detected",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        drawn,
                        "Show your hands",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

            cv2.imshow("Collect Data", drawn)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" ") and not recording:
                if last_valid_vec is not None:
                    recording = True
                    buffer.clear()
                    print(f"  Recording sequence {collected + 1}...")

    cv2.destroyAllWindows()
    print(f"\nTotal: {collected} sequences saved to {save_dir}/")


if __name__ == "__main__":
    main()
