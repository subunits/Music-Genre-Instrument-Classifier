"""
yamnet_wrapper.py  —  YAMNet classifier using TensorFlow SavedModel directly.
No tensorflow-hub dependency.

Model is downloaded once from TF Hub's CDN and cached locally as a SavedModel.
"""

import os
import csv
import urllib.request
import tarfile
import numpy as np
import tensorflow as tf

TARGET_SR   = 16000
MODEL_DIR   = os.path.join(os.path.dirname(__file__), ".yamnet_model")
MODEL_URL   = "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed"
LABELS_URL  = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

_model  = None
_labels = None


def _download_model():
    if os.path.isdir(MODEL_DIR):
        return
    print("Downloading YAMNet model (first run only)...")
    tmp = MODEL_DIR + ".tar.gz"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with tarfile.open(tmp, "r:gz") as tar:
        tar.extractall(MODEL_DIR)
    os.remove(tmp)
    print("YAMNet model ready.")


def _load_model():
    global _model, _labels
    if _model is None:
        _download_model()
        _model = tf.saved_model.load(MODEL_DIR)
    if _labels is None:
        response = urllib.request.urlopen(LABELS_URL)
        reader   = csv.DictReader(line.decode("utf-8") for line in response)
        _labels  = [row["display_name"] for row in reader]
    return _model, _labels


def _ensure_sample_rate(audio, orig_sr):
    if orig_sr == TARGET_SR:
        return audio
    import resampy
    return resampy.resample(audio, orig_sr, TARGET_SR)


def classify(audio_data, sample_rate: int):
    model, labels = _load_model()

    audio = np.asarray(audio_data, dtype=np.float32).flatten()
    audio = _ensure_sample_rate(audio, int(sample_rate))

    scores, embeddings, spectrogram = model(audio)
    mean_scores = scores.numpy().mean(axis=0)

    top_idx   = int(np.argmax(mean_scores))
    top_label = labels[top_idx]
    top_score = float(mean_scores[top_idx])

    return top_label, top_score, mean_scores.tolist()
