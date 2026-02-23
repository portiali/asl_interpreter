# Camera → MediaPipe Landmarks → LSTM

Starter project: live webcam frames, MediaPipe Pose landmarks, and a PyTorch LSTM over landmark sequences.

## Setup

```bash
cd asl_interpreter
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
## Environment
Recommended Python version: 3.10

## Run

From the project root (`camera-lstm/`):

```bash
python main.py
```
**ideally lololol but probs not teehee**
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



