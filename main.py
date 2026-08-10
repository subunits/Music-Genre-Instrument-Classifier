"""
main.py  —  Music Genre / Instrument Classifier

Usage:
    python main.py --source file   --model yamnet   --input track.mp3
    python main.py --source file   --model musicnn  --input track.wav
    python main.py --source mic    --model yamnet   --duration 5
    python main.py --source mic    --model musicnn  --duration 5
    python main.py                 # interactive prompts

Dependencies:
    pip install -r requirements.txt
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import librosa
import soundfile as sf

from helpers import preprocess_audio, extract_features, display_results


# ── Audio acquisition ──────────────────────────────────────────────────────

def load_from_file(path: str):
    """Load audio from a file — supports mp3, wav, flac, ogg."""
    audio, sr = librosa.load(path, sr=None, mono=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    return audio.astype(np.float32), sr


def record_from_mic(duration: int = 5, sr: int = 16000):
    """Record from the default microphone. Returns (audio, sr)."""
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("sounddevice not installed. Run: pip install sounddevice")

    print(f"Recording {duration} seconds from microphone...")
    audio = sd.rec(int(duration * sr), samplerate=sr,
                   channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return audio.flatten(), sr


# ── Classifier backends ────────────────────────────────────────────────────

def run_yamnet(audio: np.ndarray):
    from yamnet_wrapper import classify
    top_label, top_score, all_scores = classify(audio, sample_rate=16000)
    return top_label, top_score, all_scores, None


def run_musicnn(audio: np.ndarray):
    from essentia_wrapper import classify

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        sf.write(tmp.name, audio, 16000)
        top_label, top_score, all_scores, all_labels = classify(tmp.name)
    finally:
        os.unlink(tmp.name)

    return top_label, top_score, all_scores, all_labels


# ── Interactive prompts ────────────────────────────────────────────────────

def prompt_source():
    print("\nAudio source:")
    print("  1. Load from file")
    print("  2. Record from microphone")
    choice = input("Choice [1/2]: ").strip()
    return "file" if choice == "1" else "mic"


def prompt_model():
    print("\nClassifier:")
    print("  1. YAMNet  (521 broad AudioSet classes)")
    print("  2. Essentia MusiCNN (50 music-specific tags)")
    choice = input("Choice [1/2]: ").strip()
    return "yamnet" if choice == "1" else "musicnn"


def prompt_file():
    path = input("\nPath to audio file: ").strip()
    if not os.path.isfile(path):
        sys.exit(f"File not found: {path}")
    return path


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Music Genre / Instrument Classifier")
    parser.add_argument("--source",   choices=["file", "mic"])
    parser.add_argument("--model",    choices=["yamnet", "musicnn"])
    parser.add_argument("--input",    help="Path to audio file")
    parser.add_argument("--duration", type=int, default=5,
                        help="Recording duration in seconds")
    args = parser.parse_args()

    source = args.source or prompt_source()
    model  = args.model  or prompt_model()

    if source == "file":
        path = args.input or prompt_file()
        audio, sr = load_from_file(path)
        source_name = os.path.basename(path)
    else:
        audio, sr = record_from_mic(duration=args.duration)
        source_name = "microphone"

    print("Preprocessing audio...")
    audio = preprocess_audio(audio, sr)

    mel_db, t_vec, f_vec = extract_features(audio)

    print(f"Running {model} classifier...")
    if model == "yamnet":
        top_label, top_score, all_scores, all_labels = run_yamnet(audio)
    else:
        top_label, top_score, all_scores, all_labels = run_musicnn(audio)

    print(f"\nTop prediction: {top_label}  ({top_score*100:.1f}%)")

    display_results(mel_db, t_vec, f_vec,
                    top_label, top_score, all_scores, all_labels,
                    source_name, model)


if __name__ == "__main__":
    main()
