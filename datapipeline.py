import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download, HfApi

from src.landmarks import HolisticLandmarkExtractor, HOLISTIC_VEC_SIZE

REPO_ID = "ZahidYasinMittha/American-Sign-Language-Dataset"
DATA_DIR = Path(__file__).parent / "data"
TMP_DIR = DATA_DIR / "_tmp"

repo_root = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=["Aslense Dataset.csv"],
    local_dir=TMP_DIR,
)
df = pd.read_csv(Path(repo_root) / "Aslense Dataset.csv")

api = HfApi()
all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
filename_to_path = {Path(f).name: f for f in all_files if f.endswith(".mp4")}

df["repo_path"] = df["videos"].map(filename_to_path)
df = df.dropna(subset=["repo_path"]).reset_index(drop=True)


def extract_landmarks(video_path: Path, extractor: HolisticLandmarkExtractor):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    vectors = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vec = extractor.process_frame(frame)
            if vec is None:
                vec = np.zeros(HOLISTIC_VEC_SIZE, dtype=np.float32)
            vectors.append(vec)
    finally:
        cap.release()

    if not vectors:
        return None
    return np.stack(vectors)


with HolisticLandmarkExtractor() as extractor:
    for row in df.itertuples(index=False):
        word = row.word
        repo_path = row.repo_path
        video_id = Path(repo_path).stem
        word_dir = DATA_DIR / word
        out_path = word_dir / f"{video_id}.npz"

        if out_path.exists():
            continue

        local_video = None
        try:
            local_video = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=repo_path,
                local_dir=TMP_DIR,
            )

            landmarks = extract_landmarks(Path(local_video), extractor)
            if landmarks is None:
                print(f"{word}/{video_id}: extraction failed, skipping")
                continue

            word_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_path,
                landmarks=landmarks,
                label=word,
                length=landmarks.shape[0],
            )
            print(f"{word}/{video_id}: {landmarks.shape}")
        except Exception as e:
            print(f"{word}/{video_id}: error ({e}), skipping")
        finally:
            if local_video and os.path.exists(local_video):
                os.remove(local_video)
