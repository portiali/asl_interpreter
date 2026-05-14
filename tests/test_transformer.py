"""Tests for LandmarkTransformer in src/model.py."""

import pytest
import torch

from src.model import CLASS_LABELS, LandmarkTransformer, build_dummy_sequence
from src.landmarks import HOLISTIC_VEC_SIZE

NUM_CLASSES = len(CLASS_LABELS)
SEQ_LEN = 30


@pytest.fixture
def model():
    m = LandmarkTransformer(num_classes=NUM_CLASSES, seq_len=SEQ_LEN)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_output_shape_default_batch(model):
    x = build_dummy_sequence(batch=2, seq_len=SEQ_LEN)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_CLASSES)


def test_output_shape_batch_1(model):
    x = build_dummy_sequence(batch=1, seq_len=SEQ_LEN)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, NUM_CLASSES)


def test_output_shape_large_batch(model):
    x = build_dummy_sequence(batch=16, seq_len=SEQ_LEN)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (16, NUM_CLASSES)


def test_build_dummy_sequence_shape():
    x = build_dummy_sequence(batch=4, seq_len=SEQ_LEN)
    assert x.shape == (4, SEQ_LEN, HOLISTIC_VEC_SIZE)


# ---------------------------------------------------------------------------
# Gradient / training tests
# ---------------------------------------------------------------------------


def test_backward_pass():
    """Loss.backward() must not raise and produce non-None gradients."""
    m = LandmarkTransformer(num_classes=NUM_CLASSES, seq_len=SEQ_LEN)
    m.train()
    x = build_dummy_sequence(batch=2, seq_len=SEQ_LEN)
    out = m(x)
    loss = out.sum()
    loss.backward()
    for name, param in m.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_parameters_are_trainable(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total > 0


# ---------------------------------------------------------------------------
# Positional embedding boundary
# ---------------------------------------------------------------------------


def test_shorter_sequence_accepted(model):
    """Sequences shorter than seq_len must work (positions stay in-range)."""
    x = build_dummy_sequence(batch=2, seq_len=SEQ_LEN - 5)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_CLASSES)


def test_sequence_longer_than_seq_len_raises(model):
    """Passing more frames than seq_len should raise an IndexError because
    the positional embedding only has seq_len entries."""
    x = build_dummy_sequence(batch=2, seq_len=SEQ_LEN + 1)
    with pytest.raises((IndexError, RuntimeError)):
        with torch.no_grad():
            model(x)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_eval_mode_is_deterministic(model):
    x = build_dummy_sequence(batch=2, seq_len=SEQ_LEN)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)
