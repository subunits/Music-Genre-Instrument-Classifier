# Music Genre / Instrument Classifier
**Python pipeline — Essentia MusiCNN backend (YAMNet pending)**

---

## Quickstart (GitHub Codespaces)

1. Click the green **Code** button on the repo page
2. Select the **Codespaces** tab and click **Create codespace on main**
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the classifier:

```bash
python main.py --source file --model essentia --input LOOPsNine.mp3
```

Results are saved to `classifier_results.png` in the repo root.

---

## Local Setup

```bash
git clone https://github.com/subunits/Music-Genre-Instrument-Classifier.git
cd Music-Genre-Instrument-Classifier
pip install -r requirements.txt
python main.py
```

---

## Usage

```bash
python main.py --source file --model essentia --input track.mp3
python main.py --source mic  --model essentia --duration 5
python main.py                                # interactive prompts
```

---

## Known Issues

### YAMNet — `ModuleNotFoundError: No module named 'pkg_resources'`
`tensorflow-hub` currently fails on Python 3.12 in Codespaces due to a `pkg_resources` conflict. YAMNet is disabled until this is resolved. Use `--model essentia` in the meantime.

---

## Files

| File | Role |
|---|---|
| `main.py` | Entry point — audio I/O, model selection, pipeline orchestration |
| `helpers.py` | `preprocess_audio`, `extract_features`, `display_results` |
| `yamnet_wrapper.py` | YAMNet backend — 521 broad AudioSet classes (currently unavailable) |
| `essentia_wrapper.py` | Essentia MusiCNN backend — 50 music-specific tags |
| `requirements.txt` | Python dependencies |
| `.devcontainer/devcontainer.json` | Codespaces environment config |

---

## Model Comparison

| | YAMNet | Essentia MusiCNN |
|---|---|---|
| Classes | 521 (AudioSet) | 50 (MagnaTagATune or MSD) |
| Best for | Broad sound detection | Music-specific tags |
| Example tags | "Guitar music", "Drum kit" | "guitar", "rock", "electronic" |
| Framework | TensorFlow / TF Hub | TensorFlow / Essentia |
| Flag | `--model yamnet` (unavailable) | `--model essentia` |
| Status | ⚠️ Broken on Python 3.12 | ✅ Working |

### Essentia MusiCNN tag sets
Switch `MODEL` in `essentia_wrapper.py` to change the tag vocabulary:
- `"MTT_MusiCNN"` — MagnaTagATune (50 tags: instruments, tempo, mood)
- `"MSD_MusiCNN"` — Million Song Dataset (50 tags: genres, eras, moods)

---

## Pipeline

```
[File / Mic]
     │
     ▼
preprocess_audio()     mono → resample 16 kHz → normalize → trim silence
     │
     ▼
extract_features()     64-band Mel spectrogram (25 ms window, 10 ms hop)
     │
     ├──[--model yamnet]───▶  yamnet_wrapper.py   ⚠️ unavailable
     │
     └──[--model essentia]─▶  essentia_wrapper.py (temp .wav file path)
                                    │
     ◀──────────────────────────────┘
     top_label, top_score, all_scores
     │
     ▼
display_results()      Mel spectrogram + top-10 bar chart → classifier_results.png
```

---

## Extending

### Real-time classification
Swap the single `load_from_file` call in `main.py` for a loop using `sounddevice.InputStream` to classify audio in continuous chunks.

### Fine-tune on your own tags
See the [Essentia models documentation](https://essentia.upf.edu/models.html).
