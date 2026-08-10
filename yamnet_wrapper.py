"""
yamnet_wrapper.py  —  Python side of the Music Genre / Instrument Classifier
"""

import csv
import urllib.request
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# ── Constants ──────────────────────────────────────────────────────────────
YAMNET_URL    = "https://tfhub.dev/google/yamnet/1"
YAMNET_LABELS = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
TARGET_SR     = 16000   # YAMNet requires 16 kHz mono float32 in [-1, 1]

# ── Load model (cached after first call) ──────────────────────────────────
_model  = None
_labels = None

def _load_model():
    global _model, _labels
    if _model is None:
        _model = hub.load(YAMNET_URL)
    if _labels is None:
        response = urllib.request.urlopen(YAMNET_LABELS)
        reader   = csv.DictReader(line.decode("utf-8") for line in response)
        _labels  = [row["display_name"] for row in reader]
    return _model, _labels


# ── Optional: resample if passed a non-16k signal ──────────────────────────
def _ensure_sample_rate(audio, orig_sr):
    if orig_sr == TARGET_SR:
        return audio
    try:
        import resampy
        return resampy.resample(audio, orig_sr, TARGET_SR)
    except ImportError:
        raise RuntimeError(
            "resampy not installed. Run: pip install resampy\n"
        )


# ── Main inference ─────────────────────────────────────────────────────────
def classify(audio_data, sample_rate: int):
    """
    Parameters
    ----------
    audio_data  : numpy array, shape (N,), float32 or float64, mono
    sample_rate : int, original sample rate of audio_data

    Returns
    -------
    top_label  : str
    top_score  : float
    all_scores : list[float]  length = 521 (AudioSet classes)
    """
    model, labels = _load_model()

    audio = np.asarray(audio_data, dtype=np.float32).flatten()
    audio = _ensure_sample_rate(audio, int(sample_rate))

    # YAMNet returns per-frame scores; average across frames
    scores, embeddings, spectrogram = model(audio)
    mean_scores = scores.numpy().mean(axis=0)   # shape: (521,)

    top_idx   = int(np.argmax(mean_scores))
    top_label = labels[top_idx]
    top_score = float(mean_scores[top_idx])

    return top_label, top_score, mean_scores.tolist()


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "audio_data" in globals() and "sample_rate" in globals():
        top_label, top_score, all_scores = classify(audio_data, sample_rate)  # noqa: F821
