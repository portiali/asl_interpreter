# Camera → MediaPipe Landmarks → LSTM

Starter project: live webcam frames, MediaPipe Pose landmarks, and a PyTorch LSTM over landmark sequences.

## Setup

```bash
cd camera-lstm
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From the project root (`camera-lstm/`):

```bash
python main.py
```

- A window shows the camera feed with pose landmarks drawn.
- Once 16 frames of landmarks are buffered, the LSTM runs (untrained; class index is just a placeholder).
- Press **q** to quit.

## Structure

```
camera-lstm/
├── main.py              # Live pipeline: capture → landmarks → LSTM
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── capture.py       # OpenCV camera frame generator
    ├── landmarks.py     # MediaPipe Pose → 132-d vector per frame
    └── model.py         # LandmarkLSTM(seq_len, 132) → num_classes
```

- **Landmarks**: 33 pose keypoints × (x, y, z, visibility) = 132 dimensions per frame.
- **LSTM**: `(batch, seq_len, 132)` → logits over `num_classes` (e.g. action labels). Train with your own data and loss (e.g. cross-entropy).

## Next steps

1. Replace dummy labels with real action/gesture labels and train the LSTM (e.g. on saved landmark sequences).
2. Tune `SEQ_LEN` and LSTM `hidden_size` / `num_layers`.
3. Optionally use MediaPipe Hands or Holistic for hands + pose.
