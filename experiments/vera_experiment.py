"""
experiments/vera_experiment.py

Deliverable:
  - Capture frames from webcam
  - Run MediaPipe via landmarks.py
  - Print landmark vector shape
  - Save 3–5 sequences as .npy
  - Reload and inspect shapes

What this script does in plain English:
  It opens your webcam, and for each gesture you want to record, it:
    1. Shows you a live camera preview so you can get ready
    2. Waits for you to press SPACE when you're ready
    3. Records 30 frames of you performing the gesture
    4. For each frame, runs MediaPipe Pose to extract 132 numbers
       describing where your body landmarks are (33 points x 4 values each)
    5. Stacks those 30 frames into a (30, 132) numpy array = one "sequence"
    6. Saves it as a .npy file
  At the end, it reloads all saved files and prints their shapes to confirm.

Usage:
  Run from the repo root:
    python experiments/vera_experiment.py

  Follow the on-screen prompts:
    - Type a gesture label (e.g. 'hello') and press Enter
    - Press SPACE in the camera window to start recording
    - Perform the ASL gesture — recording stops automatically after SEQ_LEN frames
    - Repeat for each sequence
    - Press Q at any time to quit early
"""

import sys
import os
import numpy as np
import cv2

# This makes sure Python can find the src/ folder (capture.py, landmarks.py)
# even when running this script from inside the experiments/ folder.
# os.path.dirname(__file__) = this file's folder (experiments/)
# '..' = one level up = repo root, where src/ lives
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.capture import get_frames           # function that opens the webcam and yields frames one by one
from src.landmarks import LandmarkExtractor  # class that runs MediaPipe on a frame and returns landmark data

# Config
# These are the settings you can change if you want more/fewer frames or sequences

SEQ_LEN       = 30   # how many frames to record per gesture (30 frames ≈ 1-2 seconds)
NUM_SEQUENCES = 4    # how many different gesture sequences to record total

# Where to save the .npy files — creates an experiments/sequences/ folder automatically
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'sequences')
os.makedirs(SAVE_DIR, exist_ok=True)  # exist_ok=True means it won't crash if folder already exists

# Functions

def record_sequence(seq_len: int) -> np.ndarray:
    """
    Records exactly `seq_len` frames from the webcam, runs MediaPipe on each frame
    to extract body landmark positions, and returns them stacked as a numpy array.

    Returns:
        np.ndarray of shape (seq_len, 132)
        - seq_len rows = one row per frame
        - 132 columns = 33 body landmarks x 4 values (x, y, z, visibility)
    """
    frames = []  # will collect one 132-d vector per frame
    print(f"  Recording {seq_len} frames...", end='', flush=True)

    # LandmarkExtractor is used as a context manager (with block) so it
    # properly closes the MediaPipe model when we're done
    with LandmarkExtractor() as extractor:
        for success, frame in get_frames():
            # Skip if the camera failed to return a valid frame
            if not success or frame is None:
                continue

            # process_and_draw() does two things at once:
            #   1. Runs MediaPipe Pose on the frame -> returns a (132,) numpy array of landmark coords
            #   2. Draws the skeleton on a copy of the frame -> returns drawn_frame for display
            landmarks, drawn_frame = extractor.process_and_draw(frame)

            # If MediaPipe couldn't detect a pose in this frame (e.g. you moved out of frame),
            # use a vector of all zeros so we don't skip frames and mess up the sequence length
            if landmarks is None:
                landmarks = np.zeros(132, dtype=np.float32)

            frames.append(landmarks)  # add this frame's 132 numbers to our list

            # Show the frame with skeleton drawn + a counter so you can see recording progress
            cv2.putText(drawn_frame, f"Recording {len(frames)}/{seq_len}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("ASL Experiment", drawn_frame)
            cv2.waitKey(1)  # needed to actually render the window (1ms wait)

            # Stop once we've collected enough frames
            if len(frames) >= seq_len:
                break

    print(" done.")

    # np.stack turns our list of 132-d vectors into a 2D array of shape (seq_len, 132)
    # axis=0 means we stack them as rows (one row per frame)
    sequence = np.stack(frames, axis=0)
    return sequence


def wait_for_space_or_quit() -> bool:
    """
    Shows a live camera preview with the MediaPipe skeleton drawn on it.
    Waits for the user to press SPACE (start recording) or Q (quit).

    Returns:
        True  -> user pressed SPACE, ready to record
        False -> user pressed Q, exit the program
    """
    print("  Press SPACE to start recording, or Q to quit.", flush=True)

    with LandmarkExtractor() as extractor:
        for success, frame in get_frames():
            if not success or frame is None:
                continue

            # Draw the skeleton on the frame so the user can check they're in frame
            # We use _ to discard the landmark vector since we don't need it here
            _, drawn_frame = extractor.process_and_draw(frame)

            # Overlay instructions on screen
            cv2.putText(drawn_frame, "SPACE=Record  Q=Quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("ASL Experiment", drawn_frame)

            # waitKey returns the ASCII code of the key pressed (& 0xFF for Mac compatibility)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                return True   # SPACE pressed -> start recording
            if key == ord('q'):
                return False  # Q pressed -> quit

    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Print a summary of what's about to happen
    print("=" * 55)
    print(" ASL Interpreter — Vera's Capture + MediaPipe Experiment")
    print("=" * 55)
    print(f" Seq length : {SEQ_LEN} frames")
    print(f" Sequences  : {NUM_SEQUENCES}")
    print(f" Save dir   : {SAVE_DIR}")
    print("=" * 55)

    saved_paths = []  # keep track of all saved file paths so we can reload them at the end

    # Loop once per sequence (4 total)
    for i in range(NUM_SEQUENCES):

        # Ask the user to name this gesture — this becomes part of the filename
        label = input(f"\nSequence {i+1}/{NUM_SEQUENCES} — enter gesture label (e.g. 'hello'): ").strip()
        if not label:
            label = f"gesture_{i+1}"  # fallback if user just hits enter

        # Show camera preview and wait for SPACE — give the user time to get in position
        if not wait_for_space_or_quit():
            print("Quitting early.")
            break  # user pressed Q, stop the loop

        # ── Step 1-3: Record frames + run MediaPipe + collect landmark vectors
        sequence = record_sequence(SEQ_LEN)
        # sequence is now a numpy array of shape (30, 132)

        # Print the shape so we can verify it looks right
        print(f"  -> Landmark vector shape per frame : {sequence.shape[1]}")   # should be 132
        print(f"  -> Full sequence shape             : {sequence.shape}  (seq_len={SEQ_LEN}, landmark_dim=132)")

        # ── Step 4: Save the sequence as a .npy file
        # Filename format: <label>_seq<number>.npy  e.g. hello_seq1.npy
        filename = f"{label}_seq{i+1}.npy"
        filepath = os.path.join(SAVE_DIR, filename)
        np.save(filepath, sequence)  # saves the numpy array to disk
        saved_paths.append(filepath)
        print(f"  Saved -> {filepath}")

    # Close all OpenCV windows when done recording
    cv2.destroyAllWindows()

    # ── Step 5: Reload every saved .npy file and print its shape to confirm everything saved correctly
    print("\n" + "=" * 55)
    print(" Reloading saved sequences and inspecting shapes:")
    print("=" * 55)
    for path in saved_paths:
        data = np.load(path)  # loads the .npy back into a numpy array
        # Should print (30, 132) for each — 30 frames, 132 landmark values per frame
        print(f"  {os.path.basename(path):30s}  shape={data.shape}  dtype={data.dtype}")

    print("\nDone! All sequences saved to:", SAVE_DIR)


# Only run main() if this file is executed directly (not imported by another script)
if __name__ == "__main__":
    main()