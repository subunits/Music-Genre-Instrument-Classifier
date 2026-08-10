# Music Genre / Instrument Classifier
**Python pipeline — YAMNet & Essentia MusiCNN backends**

---

## Quickstart (GitHub Codespaces)

1. Click the green **Code** button on the repo page
2. Select the **Codespaces** tab and click **Create codespace on main**
3. Wait for the environment to build — dependencies install automatically
4. Run the classifier:

```bash
python main.py --source file --model musicnn --input track.mp3
python main.py --source file --model yamnet  --input track.mp3
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
python main.py --source file --model yamnet  --input track.mp3
python main.py --source file --model musicnn --input track.mp3
python main.py --source mic  --model yamnet  --duration 5
python main.py --source mic  --model musicnn --duration 5
python main.py                               # interactive prompts
```

---

## Files

| File | Role |
|---|---|
| `main.py` | Entry point — audio I/O, model selection, pipeline orchestration |
| `helpers.py` | `preprocess_audio`, `extract_features`, `display_results` |
| `yamnet_wrapper.py` | YAMNet backend — 521 broad AudioSet classes |
| `essentia_wrapper.py` | Essentia MusiCNN backend — 50 music-specific tags |
| `requirements.txt` | Python dependencies |
| `.devcontainer/devcontainer.json` | Codespaces environment config |

---

## Model Comparison

| | YAMNet | Essentia MusiCNN |
|---|---|---|
| Classes | 521 (AudioSet) | 50 (MagnaTagATune or MSD) |
| Best for | Broad sound detection | Music-specific tags |
| Example tags | "Guitar music", "Drum kit" | "guitar", "rock", "female vocals" |
| Framework | TensorFlow / TF Hub | TensorFlow / Essentia |
| Speed | ~1–2 s | ~2–4 s |

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
     ├──[yamnet]──▶  yamnet_wrapper.py    (raw audio array)
     │
     └──[musicnn]─▶  essentia_wrapper.py  (temp .wav file path)
                           │
     ◀─────────────────────┘
     top_label, top_score, all_scores
     │
     ▼
display_results()      Mel spectrogram + top-10 bar chart → classifier_results.png
```

---

## Extending

### Real-time classification
Swap the single `load_from_file` call in `main.py` for a loop using `sounddevice.InputStream` to classify audio in continuous chunks.

### Fine-tune MusiCNN on your own tags
See the [Essentia models documentation](https://essentia.upf.edu/models.html).
