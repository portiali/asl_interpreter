"""
Live ASL interpreter web app.

Pipeline:
    webcam -> MediaPipe Holistic landmarks (147-d / frame) -> 120-frame rolling
    buffer -> LandmarkTransformer -> emitted sign words -> OpenAI streaming
    translation -> SSE to the browser caption bar.

Run:
    python app.py
Then open http://localhost:5000
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, stream_with_context
from openai import OpenAI

from src.capture import get_frames
from src.landmarks import HOLISTIC_VEC_SIZE, HolisticLandmarkExtractor
from src.model import load_label_map, load_model

load_dotenv()

# --- Configuration ---
SILENCE_SECONDS = 3.5         # how long the buffer must sit idle before translating
MIN_SIGNS_TO_TRANSLATE = 1
JPEG_QUALITY = 65
CAMERA_W, CAMERA_H = 640, 480
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# --- Transformer inference config ---
TFM_CHECKPOINT = os.environ.get("TFM_CHECKPOINT", "models/best_model_transformer.pt")
TFM_LABEL_MAP_PATHS = ["models/label_map.json", "label_map.json", "data/label_map.json"]
TFM_CONF_THRESHOLD = 0.10    # min softmax prob to count as a vote (1740-class model is diffuse)
TFM_INFER_EVERY = 6          # run inference every N frames (~5 Hz at 30 fps)
TFM_CONSISTENT_VOTES = 2     # need this many consecutive same predictions to emit
TFM_COOLDOWN_SEC = 2.0       # don't repeat the SAME sign within this window
TFM_GLOBAL_COOLDOWN = 0.6    # min gap between any two emissions
TFM_MIN_HAND_FRAMES = 20     # need at least this many frames with hands in the 120-frame buffer
TFM_DEBUG = False            # print top prediction every inference

app = Flask(__name__)


# --- Shared state ---
@dataclass
class AppState:
    latest_jpeg: Optional[bytes] = None
    fps: float = 0.0
    pending_signs: list[str] = field(default_factory=list)
    last_sign_time: float = 0.0
    listeners: list["queue.Queue[dict]"] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


state = AppState()
openai_client: Optional[OpenAI] = None
tfm_model = None                 # LandmarkTransformer (eval mode, on CPU/GPU)
tfm_seq_len: int = 120           # filled in at load time
tfm_labels: dict[int, str] = {}  # idx -> word, from label_map.json
tfm_device: Optional[torch.device] = None


def _broadcast(event: dict) -> None:
    with state.lock:
        listeners = list(state.listeners)
    for q in listeners:
        try:
            q.put_nowait(event)
        except queue.Full:
            pass


def _emit_sign(word: str) -> None:
    now = time.time()
    with state.lock:
        state.pending_signs.append(word)
        state.last_sign_time = now
    _broadcast({"type": "sign", "word": word, "confidence": 1.0})


# --- Capture thread (visuals + optional transformer inference) ---
def _capture_loop() -> None:
    has_tfm = tfm_model is not None
    print(
        "[capture] starting MediaPipe overlay"
        + (
            f" + Transformer ({len(tfm_labels)} classes, seq_len={tfm_seq_len})"
            if has_tfm else " (no inference)"
        )
    )
    prev_time = time.time()

    # Rolling buffer of holistic vectors (147-d each), kept at tfm_seq_len.
    # `hand_flags` tracks frames where at least one hand was actually detected.
    frame_buf: list[np.ndarray] = []
    hand_flags: list[bool] = []
    frames_seen = 0
    last_emit_per_word: dict[str, float] = {}
    last_global_emit = 0.0
    consec_word: Optional[str] = None
    consec_count = 0

    with HolisticLandmarkExtractor(model_complexity=0) as extractor:
        for ok, frame in get_frames(camera_id=0, width=CAMERA_W, height=CAMERA_H):
            if not ok or frame is None:
                break

            vec, drawn = extractor.process_and_draw(frame)

            if has_tfm:
                has_hand = vec is not None
                frame_buf.append(
                    vec if has_hand
                    else np.zeros(HOLISTIC_VEC_SIZE, dtype=np.float32)
                )
                hand_flags.append(has_hand)
                if len(frame_buf) > tfm_seq_len:
                    frame_buf.pop(0)
                    hand_flags.pop(0)
                frames_seen += 1

                enough_hands = sum(hand_flags) >= TFM_MIN_HAND_FRAMES
                if (
                    len(frame_buf) == tfm_seq_len
                    and frames_seen % TFM_INFER_EVERY == 0
                    and enough_hands
                ):
                    word, conf = _tfm_predict(frame_buf)
                    now = time.time()
                    if TFM_DEBUG and word is not None:
                        print(f"[tfm] top={word!r} conf={conf:.3f} hands={sum(hand_flags)}/{tfm_seq_len}")

                    # Vote-based smoothing: same class N times in a row,
                    # each above threshold, before we emit.
                    if word is not None and conf >= TFM_CONF_THRESHOLD:
                        if word == consec_word:
                            consec_count += 1
                        else:
                            consec_word = word
                            consec_count = 1
                    else:
                        consec_word = None
                        consec_count = 0

                    if (
                        consec_word is not None
                        and consec_count >= TFM_CONSISTENT_VOTES
                        and (now - last_global_emit) >= TFM_GLOBAL_COOLDOWN
                        and (now - last_emit_per_word.get(consec_word, 0.0)) >= TFM_COOLDOWN_SEC
                    ):
                        _emit_sign(consec_word)
                        last_emit_per_word[consec_word] = now
                        last_global_emit = now
                        cv2.putText(
                            drawn, f"{consec_word} {conf:.2f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2,
                        )
                        # After emitting, drain the buffer so the next sign
                        # starts from a clean window.
                        frame_buf.clear()
                        hand_flags.clear()
                        consec_word = None
                        consec_count = 0

            now = time.time()
            dt = now - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = now

            drawn = _draw_fps(drawn, fps)
            ok2, jpeg = cv2.imencode(
                ".jpg", drawn, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if ok2:
                with state.lock:
                    state.latest_jpeg = jpeg.tobytes()
                    state.fps = fps


def _tfm_predict(frame_buf: list[np.ndarray]) -> tuple[Optional[str], float]:
    if tfm_model is None or not tfm_labels:
        return None, 0.0
    arr = np.stack(frame_buf, axis=0).astype(np.float32)  # (T, 147)
    x = torch.from_numpy(arr).unsqueeze(0)  # (1, T, 147)
    if tfm_device is not None:
        x = x.to(tfm_device)
    with torch.no_grad():
        logits = tfm_model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs))
        word = tfm_labels.get(idx, f"class_{idx}")
        return word, float(probs[idx])


def _draw_fps(frame, fps: float):
    text = f"{fps:.0f} FPS"
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        frame, text, (w - tw - 12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )
    return frame


# --- Translation thread ---
def _translate_loop() -> None:
    while True:
        time.sleep(0.2)
        with state.lock:
            signs = list(state.pending_signs)
            last_t = state.last_sign_time
        if (
            len(signs) >= MIN_SIGNS_TO_TRANSLATE
            and time.time() - last_t >= SILENCE_SECONDS
        ):
            with state.lock:
                state.pending_signs = []
            _stream_translation(signs)


def _stream_translation(signs: list[str]) -> None:
    gloss = " ".join(signs)
    _broadcast({"type": "translating", "gloss": gloss})

    if openai_client is None:
        fallback = f"[no OPENAI_API_KEY] gloss: {gloss}"
        for ch in fallback:
            _broadcast({"type": "token", "text": ch})
            time.sleep(0.01)
        _broadcast({"type": "sentence_done", "text": fallback})
        return

    try:
        stream = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You translate American Sign Language gloss into natural, "
                        "grammatical English. Output ONLY the English sentence — "
                        "no quotes, no commentary. ASL gloss omits articles and uses "
                        "topic-comment order; restore them. If the gloss is a single "
                        "word, output a complete short utterance."
                    ),
                },
                {"role": "user", "content": f"Gloss: {gloss}"},
            ],
        )
        full = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full.append(delta)
                _broadcast({"type": "token", "text": delta})
        _broadcast({"type": "sentence_done", "text": "".join(full)})
    except Exception as e:
        msg = f"[openai error: {e}]"
        _broadcast({"type": "token", "text": msg})
        _broadcast({"type": "sentence_done", "text": msg})


# --- Routes ---
@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/video_feed")
def video_feed() -> Response:
    def gen():
        boundary = b"--frame\r\n"
        while True:
            with state.lock:
                jpeg = state.latest_jpeg
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(1 / 30)

    return Response(
        stream_with_context(gen()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/captions")
def captions() -> Response:
    q: queue.Queue[dict] = queue.Queue(maxsize=128)
    with state.lock:
        state.listeners.append(q)
    q.put_nowait({"type": "hello"})

    def gen():
        try:
            while True:
                try:
                    event = q.get(timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with state.lock:
                if q in state.listeners:
                    state.listeners.remove(q)

    return Response(stream_with_context(gen()), mimetype="text/event-stream")


@app.route("/translate_now", methods=["POST"])
def translate_now() -> Response:
    """Force-translate the current buffer (skip the silence wait)."""
    with state.lock:
        state.last_sign_time = 0.0  # makes the watcher fire immediately
    return jsonify({"ok": True})


def _start_background_threads() -> None:
    threading.Thread(target=_capture_loop, daemon=True).start()
    threading.Thread(target=_translate_loop, daemon=True).start()


if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY"):
        globals()["openai_client"] = OpenAI()
        print(f"[app] OpenAI ready ({OPENAI_MODEL})")
    else:
        print("[app] OPENAI_API_KEY not set — translation will echo gloss")

    if os.path.exists(TFM_CHECKPOINT):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, num_classes, seq_len = load_model(TFM_CHECKPOINT, device)
            labels, label_src = load_label_map(TFM_LABEL_MAP_PATHS)
            if not labels:
                print(
                    f"[app] WARNING: no label_map.json found in {TFM_LABEL_MAP_PATHS} —"
                    " predictions will show as class_<idx>"
                )
                labels = {i: f"class_{i}" for i in range(num_classes)}
            elif len(labels) != num_classes:
                print(
                    f"[app] WARNING: label_map has {len(labels)} entries but model has"
                    f" {num_classes} classes ({label_src})"
                )
            globals()["tfm_model"] = model
            globals()["tfm_labels"] = labels
            globals()["tfm_seq_len"] = seq_len
            globals()["tfm_device"] = device
            print(
                f"[app] Transformer loaded from {TFM_CHECKPOINT} on {device} — "
                f"{num_classes} classes, seq_len={seq_len}, labels={label_src}"
            )
        except Exception as e:
            print(f"[app] failed to load transformer checkpoint ({TFM_CHECKPOINT}): {e}")
    else:
        print(
            f"[app] no transformer checkpoint at {TFM_CHECKPOINT} —"
            " sign detection disabled"
        )

    _start_background_threads()
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
