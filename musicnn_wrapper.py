import os
import urllib.request
import numpy as np
import essentia.standard as es

MODEL = "MSD_MusiCNN"   # or "MTT_MusiCNN"
TARGET_SR = 16000

# Direct download links for official Essentia MusiCNN TensorFlow models
MODEL_URLS = {
    "MSD_MusiCNN": "https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.pb",
    "MTT_MusiCNN": "https://essentia.upf.edu/models/autotagging/mtt/mtt-musicnn-1.pb",
}

# (Keep your MTT_TAGS, MSD_TAGS, and TAG_MAP defined above as-is)

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

def classify(audio_path: str, model: str = MODEL):
    labels = TAG_MAP.get(model, MSD_TAGS)
    graph_file = get_model_path(model)

    # Load audio at 16 kHz mono
    loader = es.MonoLoader(filename=audio_path, sampleRate=TARGET_SR)
    audio = loader()

    # Run MusiCNN predictor using the local .pb graph file
    predictor = es.TensorflowPredictMusiCNN(
        graphFilename=graph_file,
        output="model/Sigmoid"
    )
    activations = predictor(audio)

    mean_scores = np.array(activations).mean(axis=0)
    top_idx = int(np.argmax(mean_scores))
    top_label = labels[top_idx]
    top_score = float(mean_scores[top_idx])

    return top_label, top_score, mean_scores.tolist(), labels
