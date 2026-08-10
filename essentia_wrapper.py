"""
essentia_wrapper.py — Music-specific tagger using Essentia's MusiCNN model
"""

import os
import urllib.request
import numpy as np
import essentia.standard as es

MODEL = "MSD_MusiCNN"   # swap to "MTT_MusiCNN" for instrument/mood tags
TARGET_SR = 16000

# ── Tag vocabularies ───────────────────────────────────────────────────────

MTT_TAGS = [
    "guitar", "classical", "slow", "techno", "strings", "drums", "electronic",
    "rock", "fast", "piano", "ambient", "beat", "violin", "vocal", "synth",
    "female", "indian", "opera", "male", "singing", "vocals", "no vocals",
    "harpsichord", "loud", "quiet", "flute", "woman", "male vocal",
    "no vocal", "pop", "soft", "sitar", "solo", "man", "classic", "choir",
    "voice", "new age", "dance", "male voice", "female vocal", "beats",
    "harp", "cello", "no voice", "weird", "country", "metal", "female voice",
    "choral",
]

MSD_TAGS = [
    "rock", "pop", "alternative", "indie", "electronic", "female vocalists",
    "dance", "00s", "alternative rock", "jazz", "beautiful", "metal",
    "chillout", "male vocalists", "classic rock", "soul", "indie rock",
    "mellow", "electronica", "80s", "folk", "90s", "chill", "instrumental",
    "punk", "oldies", "blues", "hard rock", "ambient", "acoustic",
    "experimental", "female vocalist", "guitar", "hip-hop", "70s",
    "party", "country", "easy listening", "sexy", "catchy", "funk",
    "electro", "heavy metal", "progressive rock", "60s", "rnb", "indie pop",
    "sad", "house", "happy",
]

TAG_MAP = {
    "MSD_MusiCNN": MSD_TAGS,
    "MTT_MusiCNN": MTT_TAGS,
}

MODEL_URLS = {
    "MSD_MusiCNN": "https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.pb",
    "MTT_MusiCNN": "https://essentia.upf.edu/models/autotagging/mtt/mtt-musicnn-1.pb",
}

# ── Model Download Helper ──────────────────────────────────────────────────

def get_model_path(model_name: str) -> str:
    """Download the model .pb file if not already cached locally."""
    filename = f"{model_name.lower().replace('_', '-')}-1.pb"
    if not os.path.exists(filename):
        url = MODEL_URLS.get(model_name)
        if not url:
            raise ValueError(f"Unknown model: {model_name}")
        print(f"Downloading Essentia model graph ({filename})...")
        urllib.request.urlretrieve(url, filename)
    return filename

# ── Inference ──────────────────────────────────────────────────────────────

def classify(audio_path: str, model: str = MODEL):
    """
    Parameters
    ----------
    audio_path : str   path to a .wav file
    model      : str   "MSD_MusiCNN" or "MTT_MusiCNN"

    Returns
    -------
    top_label  : str
    top_score  : float
    all_scores : list[float]
    all_labels : list[str]
    """
    labels = TAG_MAP.get(model, MSD_TAGS)
    graph_file = get_model_path(model)

    # Load audio at 16 kHz mono
    loader = es.MonoLoader(filename=audio_path, sampleRate=TARGET_SR)
    audio = loader()

    # Run MusiCNN predictor using local .pb graph file
    predictor = es.TensorflowPredictMusiCNN(
        graphFilename=graph_file,
        output="model/Sigmoid"
    )
    activations = predictor(audio)   # shape: (frames, n_tags)

    mean_scores = np.array(activations).mean(axis=0)
    top_idx = int(np.argmax(mean_scores))
    top_label = labels[top_idx]
    top_score = float(mean_scores[top_idx])

    return top_label, top_score, mean_scores.tolist(), labels
