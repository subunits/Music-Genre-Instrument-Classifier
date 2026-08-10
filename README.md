# Music Genre / Instrument Classifier
**MATLAB + Python scaffold — YAMNet & musicnn backends**

---

## Files

| File | Role |
|---|---|
| `main.m` | Entry point — audio I/O, model selection, pipeline orchestration |
| `helpers.m` | `preprocess_audio`, `extract_features`, `display_results` |
| `yamnet_wrapper.py` | YAMNet backend — 521 broad AudioSet classes |
| `musicnn_wrapper.py` | musicnn backend — 50 music-specific tags (genres, moods, instruments) |
| `README.md` | This file |

---

## Setup

### 1. Python dependencies

**For YAMNet:**
```bash
pip install tensorflow tensorflow-hub numpy resampy
```

**For musicnn:**
```bash
pip install musicnn
```

Install both if you want to use either model.

### 2. Configure MATLAB's Python interpreter
```matlab
pe = pyenv("Version", "3.9")   % adjust to your Python version
pe.Executable                  % confirm it points to the right environment
```

### 3. MATLAB toolboxes required
- **Audio Toolbox** (`melSpectrogram`, `audioDeviceReader`, `audioread`)

---

## Usage

```matlab
addpath(fileparts(mfilename("fullpath")));  % add helpers to path

main()                   % prompted: source + model
main("file")             % file picker, then prompted for model
main("mic")              % record 5 s from mic, then prompted for model
main("file", "yamnet")   % file + YAMNet
main("file", "musicnn")  % file + musicnn
main("mic",  "musicnn")  % mic  + musicnn
```

---

## Model Comparison

| | YAMNet | musicnn |
|---|---|---|
| Classes | 521 (AudioSet) | 50 (MagnaTagATune or MSD) |
| Best for | Broad sound detection | Music-specific tags |
| Example tags | "Guitar music", "Drum kit" | "guitar", "rock", "female vocals" |
| Input | Raw audio array (16 kHz) | .wav file path |
| Framework | TensorFlow / TF Hub | PyTorch (musicnn) |
| Speed | ~1–2 s | ~2–4 s |

### musicnn tag sets
Switch `MODEL` in `musicnn_wrapper.py` to change the tag vocabulary:
- `"MTT_musicnn"` — MagnaTagATune (50 tags: instruments, tempo, mood)
- `"MSD_musicnn"` — Million Song Dataset (50 tags: genres, eras, moods)

---

## Pipeline

```
[Mic / File]
     │
     ▼
preprocess_audio()     mono → resample 16 kHz → normalize → trim silence
     │
     ▼
extract_features()     64-band Mel spectrogram (25 ms window, 10 ms hop)
     │
     ├──[yamnet]──▶  yamnet_wrapper.py   (raw audio array via py.numpy.array)
     │                     │
     └──[musicnn]─▶  write temp .wav  →  musicnn_wrapper.py  (file path)
                           │
     ◀─────────────────────┘
     top_label, top_score, all_scores
     │
     ▼
display_results()      Mel spectrogram + top-10 horizontal bar chart
```

---

## Extending

### Go real-time (continuous mic)
```matlab
while true
    audio = recorder();
    audio = preprocess_audio(audio, fs);
    [melSpec, tVec, fVec] = extract_features(audio, fs);
    % ... classify and refresh display
    drawnow;
end
```

### Fine-tune musicnn on your own tags
musicnn supports transfer learning — see the
[musicnn documentation](https://github.com/jordipons/musicnn-training)
for training on a custom dataset.
