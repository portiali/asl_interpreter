import pandas as pd
import os
import cv2
import mediapipe as mp
import numpy as np


df = pd.read_csv("dataset/Aslense Dataset.csv")
print(df.head())

# Extract keypoints from one video using a function
mp_holistic = mp.solutions.holistic

def extract_keypoints(frame, holistic):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    keypoints = []

    # Pose (33 × 3 = 99)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*99)

    # Left hand (21 × 3 = 63)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*63)

    # Right hand (21 × 3 = 63)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*63)

    return np.array(keypoints)

# Normalization function
def normalize_sequence(seq, length=30):
    if len(seq) > length:
        return seq[:length]
    elif len(seq) < length:
        padding = np.zeros((length - len(seq), seq.shape[1]))
        return np.vstack((seq, padding))
    return seq

# Label -> index mapping
labels = sorted(df['word'].unique())
label_map = {label: idx for idx, label in enumerate(labels)}

# print("Label map:", label_map)

np.save("data/label_map.npy", label_map)

# Load 20 videos for now (using OpenCV)
for i in range(20):
    row = df.iloc[i]
    video_name = row['videos']
    label = row['word']

    print(f"\nProcessing {video_name} ({label})")

    video_path = None

    for root, dirs, files in os.walk("dataset"):
        if video_name in files:
            video_path = os.path.join(root, video_name)
            break

    if video_path is None:
        print(f"Video {video_name} not found, skipping...")
        continue

    cap = cv2.VideoCapture(video_path)

    print("Opened:", cap.isOpened())

    # Convert video -> sequence (run it on video)
    sequence = []

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = extract_keypoints(frame, holistic)
            sequence.append(keypoints)

    cap.release()

    sequence = np.array(sequence)
    print("Sequence shape:", sequence.shape)

    # Trimming to 147 features instead of 225
    sequence = sequence[:, :147]
    print("After feature trim:", sequence.shape)

    # Apply the normalization
    sequence = normalize_sequence(sequence, 30)
    print("Final shape:", sequence.shape)

    # Saving as .npy
    save_dir = f"data/{label}"
    os.makedirs(save_dir, exist_ok=True)

    existing_files = len(os.listdir(save_dir))
    np.save(f"{save_dir}/Seq_{existing_files}.npy", sequence)

    print(f"Saved file to {save_dir}/Seq_{existing_files}.npy")
