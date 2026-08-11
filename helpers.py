"""
helpers.py  —  Audio preprocessing, feature extraction, and display.
Replaces helpers.m for the pure-Python / Codespaces pipeline.

Dependencies:
    pip install librosa numpy matplotlib sounddevice
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

TARGET_SR = 16000   # Both YAMNet and Essentia MusiCNN expect 16 kHz


# ── 1. Preprocess ──────────────────────────────────────────────────────────

def preprocess_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Prepare raw audio for the classifier.

    Steps:
      - Mix stereo down to mono
      - Resample to 16 kHz
      - Normalize amplitude to [-1, 1]
      - Trim leading/trailing silence

    Parameters
    ----------
    audio : np.ndarray  shape (N,) or (N, channels)
    sr    : int         original sample rate

    Returns
    -------
    audio : np.ndarray  mono, 16 kHz, normalized, silence-trimmed
    """
    # Mono mixdown
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    # Normalize
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=40)

    return audio.astype(np.float32)


# ── 2. Feature extraction ──────────────────────────────────────────────────

def extract_features(audio: np.ndarray, sr: int = TARGET_SR):
    """
    Compute a log-power Mel spectrogram for visualisation.

    Parameters
    ----------
    audio : np.ndarray  mono float32 at TARGET_SR
    sr    : int         sample rate (default 16000)

    Returns
    -------
    mel_db  : np.ndarray  [n_mels × n_frames] log-power spectrogram
    t_vec   : np.ndarray  time axis in seconds
    f_vec   : np.ndarray  centre frequencies of Mel bands (Hz)
    """
    N_MELS     = 64
    N_FFT      = 512          # ~32 ms at 16 kHz
    HOP_LENGTH = 160          # 10 ms hop
    WIN_LENGTH = 400          # 25 ms window

    mel = librosa.feature.melspectrogram(
        y          = audio,
        sr         = sr,
        n_fft      = N_FFT,
        hop_length = HOP_LENGTH,
        win_length = WIN_LENGTH,
        n_mels     = N_MELS,
        fmin       = 0,
        fmax       = sr // 2,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    n_frames = mel_db.shape[1]
    t_vec    = librosa.frames_to_time(
                   np.arange(n_frames), sr=sr, hop_length=HOP_LENGTH)
    f_vec    = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=sr // 2)

    return mel_db, t_vec, f_vec


# ── 3. Display ─────────────────────────────────────────────────────────────

def display_results(mel_db, t_vec, f_vec,
                    top_label, top_score, all_scores, all_labels,
                    source_name, model_name):
    """
    Plot the Mel spectrogram and top-10 classification bar chart.

    Parameters
    ----------
    mel_db      : np.ndarray  [n_mels × n_frames]
    t_vec       : np.ndarray  time axis
    f_vec       : np.ndarray  frequency axis
    top_label   : str
    top_score   : float       in [0, 1]
    all_scores  : list[float]
    all_labels  : list[str] or None   (None for YAMNet → uses class indices)
    source_name : str
    model_name  : str         "yamnet" or "essentia"
    """
    TOP_N = min(10, len(all_scores))
    sorted_idx    = np.argsort(all_scores)[::-1][:TOP_N]
    sorted_scores = np.array(all_scores)[sorted_idx] * 100

    if all_labels:
        tick_labels = [all_labels[i] for i in sorted_idx]
    else:
        tick_labels = [f"Class {i}" for i in sorted_idx]

    fig = plt.figure(figsize=(11, 8))
    fig.suptitle("Music Genre / Instrument Classifier", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 1, hspace=0.45)

    # -- Mel spectrogram --
    ax1 = fig.add_subplot(gs[0])
    img = librosa.display.specshow(
        mel_db, x_coords=t_vec, y_coords=f_vec,
        x_axis="time", y_axis="mel",
        sr=TARGET_SR, hop_length=160,
        cmap="inferno", ax=ax1
    )
    fig.colorbar(img, ax=ax1, format="%+2.0f dB")
    ax1.set_title(f"Mel Spectrogram — {source_name}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")

    # -- Top-N bar chart --
    ax2   = fig.add_subplot(gs[1])
    y_pos = np.arange(TOP_N)
    colors = ["#2176b5"] + ["#8dc4e8"] * (TOP_N - 1)   # winner highlighted

    ax2.barh(y_pos, sorted_scores, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(tick_labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Score (%)")
    ax2.set_xlim(0, 100)
    ax2.set_title(f"[{model_name.upper()}]  Top: {top_label}  ({top_score*100:.1f}%)")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    plt.savefig("classifier_results.png", dpi=150, bbox_inches="tight")
    print("Plot saved to classifier_results.png")
    plt.show()
