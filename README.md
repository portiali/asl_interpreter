# Camera → MediaPipe Landmarks → Transformer

Live ASL interpreter: webcam frames, MediaPipe Holistic landmarks, and a PyTorch transformer over landmark sequences. Predicted signs feed into OpenAI for English translation, served over a Flask web app.

## Setup

```bash
cd asl_interpreter
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment
Recommended Python version: 3.10

Optional `.env`:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
TFM_CHECKPOINT=models/best_model_transformer.pt
```

## Run

From the project root (`asl_interpreter/`):

```bash
python app.py
```

Then open http://localhost:5000

* Browser shows the live camera feed with MediaPipe landmarks drawn.
* Detected signs appear as chips in the side panel.
* After a short pause, the gloss is sent to OpenAI and the English sentence streams into the caption bar.
* "Translate Now" button forces a translation without waiting for the silence gap.

## Structure

```
asl_interpreter/
├── app.py                  # Flask app: capture → landmarks → transformer → translation → SSE
├── requirements.txt
├── README.md
├── models/
│   ├── best_model_transformer.pt
│   └── label_map.json      # idx → word for the trained transformer
├── scripts/                # Offline tools (not used at runtime)
│   ├── train.py            # Trainer for the transformer
│   ├── collect_data.py     # Webcam recorder for new sign sequences
│   ├── datapipeline.py     # Hugging Face dataset → landmark sequences
│   └── build_label_map.py  # Reconstruct label_map.json from CSV
├── templates/
│   └── index.html
├── static/
│   ├── app.js
│   └── style.css
└── src/
    ├── __init__.py
    ├── capture.py          # OpenCV camera frame generator
    ├── landmarks.py        # MediaPipe Holistic → 147-d vector per frame
    ├── model.py            # LandmarkTransformer + checkpoint loader
    └── smoothing.py        # Vote/cooldown helpers
```

Scripts are run from the project root, e.g. `python scripts/train.py --data_dir data`.
