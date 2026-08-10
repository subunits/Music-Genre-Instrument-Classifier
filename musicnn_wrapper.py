"""
musicnn_wrapper.py  —  Music-specific tagger using musicnn
Called from MATLAB via:
    pyrunfile("musicnn_wrapper.py", ["top_label","top_score","all_scores","all_labels"],
              audio_path=..., topn=...)

Dependencies (install once):
    pip install musicnn

Notes:
  - musicnn requires a file path (not raw audio array), so MATLAB must first
    write audio to a temp .wav file — see main.m for how this is handled.
  - Two models are available:
      "MTT_musicnn"  — trained on MagnaTagATune (music tags: genre, mood, instrument)
      "MSD_musicnn"  — trained on Million Song Dataset (broader music taxonomy)
  - Output scores are in [0, 1]; they are not a probability distribution
    (multiple tags can score high simultaneously).

Outputs (returned to MATLAB):
    top_label   str         — highest-scoring tag
    top_score   float       — its score in [0, 1]
    all_scores  list[float] — scores for every tag
    all_labels  list[str]   — tag names matching all_scores
"""

import numpy as np
from musicnn.tagger import top_tags
from musicnn.extractor import extractor

# ── Configuration ─────────────────────────────────────────────────────────
MODEL   = "MTT_musicnn"   # swap to "MSD_musicnn" for Million Song Dataset tags
TOP_N   = 10              # how many tags to return as "top"

# MagnaTagATune tag list (50 tags, same order as model output)
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

TAG_MAP = {"MTT_musicnn": MTT_TAGS, "MSD_musicnn": MSD_TAGS}


# ── Main inference ─────────────────────────────────────────────────────────
def classify(audio_path: str, model: str = MODEL, topn: int = TOP_N):
    """
    Parameters
    ----------
    audio_path : str   path to a .wav file (written by MATLAB)
    model      : str   "MTT_musicnn" or "MSD_musicnn"
    topn       : int   number of top tags to surface

    Returns
    -------
    top_label  : str
    top_score  : float
    all_scores : list[float]
    all_labels : list[str]
    """
    labels = TAG_MAP.get(model, MTT_TAGS)

    # extractor returns taggram (frames × tags) + stats dict
    taggram, stats, _ = extractor(audio_path, model=model, extract_features=False)

    # Average across time frames → single score per tag
    mean_scores = np.array(taggram).mean(axis=0)   # shape: (50,) for MTT

    top_idx   = int(np.argmax(mean_scores))
    top_label = labels[top_idx]
    top_score = float(mean_scores[top_idx])

    return top_label, top_score, mean_scores.tolist(), labels


# ── Entry point ───────────────────────────────────────────────────────────
# MATLAB passes `audio_path` (string) and optionally `topn` (int).
_topn = int(topn) if "topn" in dir() else TOP_N          # noqa: F821
top_label, top_score, all_scores, all_labels = classify(  # noqa: F821
    str(audio_path), topn=_topn                           # noqa: F821
)
