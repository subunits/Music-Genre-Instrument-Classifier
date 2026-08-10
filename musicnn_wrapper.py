"""
musicnn_wrapper.py  —  Music-specific tagger using Essentia's MusiCNN model
Replaces the original musicnn library (abandoned, numpy conflict).

Called from main.py via:
    from musicnn_wrapper import classify

Dependencies:
    pip install essentia-tensorflow

Models available:
    "MSD_MusiCNN"   — Million Song Dataset (200 tags: genres, moods, instruments)
    "MTT_MusiCNN"   — MagnaTagATune (50 tags: instruments, tempo, mood)

Outputs:
    top_label   str         — highest-scoring tag
    top_score   float       — its score in [0, 1]
    all_scores  list[float] — scores for every tag
    all_labels  list[str]   — tag names matching all_scores
"""

import numpy as np

MODEL     = "MSD_MusiCNN"   # swap to "MTT_MusiCNN" for instrument/mood tags
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

TAG_MAP = {"MSD_MusiCNN": MSD_TAGS, "MTT_MusiCNN": MTT_TAGS}


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
    import essentia.standard as es

    labels = TAG_MAP.get(model, MSD_TAGS)

    # Load audio at 16 kHz mono
    loader    = es.MonoLoader(filename=audio_path, sampleRate=TARGET_SR)
    audio     = loader()

    # Run MusiCNN via Essentia's TensorflowPredictMusiCNN
    predictor = es.TensorflowPredictMusiCNN(
        graphFilename=es.tensorflow_predict_musicnn_model(model),
        output="model/Sigmoid"
    )
    activations = predictor(audio)   # shape: (frames, n_tags)

    mean_scores = np.array(activations).mean(axis=0)
    top_idx     = int(np.argmax(mean_scores))
    top_label   = labels[top_idx]
    top_score   = float(mean_scores[top_idx])

    return top_label, top_score, mean_scores.tolist(), labels
