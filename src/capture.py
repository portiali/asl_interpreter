"""
Live frame capture from the default camera using OpenCV.
"""
import cv2
from typing import Generator, Optional


def get_frames(
    camera_id: int = 0,
    width: int = 640,
    height: int = 480,
    mirror: bool = True,
) -> Generator[tuple[bool, Optional["cv2.Mat"]], None, None]:
    """
    Yield (success, frame) from the default camera.
    Press 'q' to stop (handled by caller if they display the frame).
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        yield False, None
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                yield False, None
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            yield True, frame
    finally:
        cap.release()
