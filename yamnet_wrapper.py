# ── Entry point when called via pyrunfile or standalone ────────────────────
if __name__ == "__main__":
    # If run directly or via pyrunfile where globals exist
    if "audio_data" in globals() and "sample_rate" in globals():
        top_label, top_score, all_scores = classify(audio_data, sample_rate)  # noqa: F821
