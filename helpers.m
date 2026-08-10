% =========================================================
% helpers.m  —  Preprocessing, feature extraction, display
% Call these from main.m or use independently.
% =========================================================


function audio = preprocess_audio(audio, fs)
% PREPROCESS_AUDIO  Prepare raw audio for the classifier.
%
%   - Mix stereo down to mono
%   - Resample to 16 kHz (YAMNet's expected rate)
%   - Normalize amplitude to [-1, 1]
%   - Trim leading/trailing silence

    TARGET_FS    = 16000;
    SILENCE_THRESH = 0.01;   % RMS threshold for silence trimming

    % Mono mixdown
    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end

    % Resample
    if fs ~= TARGET_FS
        audio = resample(audio, TARGET_FS, fs);
    end

    % Normalize
    peak = max(abs(audio));
    if peak > 0
        audio = audio / peak;
    end

    % Trim silence from both ends
    rms_env   = movmean(audio .^ 2, round(TARGET_FS * 0.02));
    active    = rms_env > SILENCE_THRESH ^ 2;
    first_idx = find(active, 1, "first");
    last_idx  = find(active, 1, "last");
    if ~isempty(first_idx)
        audio = audio(first_idx:last_idx);
    end
end


function [melSpec, tVec, fVec] = extract_features(audio, fs)
% EXTRACT_FEATURES  Compute a Mel spectrogram for visualisation.
%
%   Returns:
%     melSpec  — [numBands × numFrames] log-power Mel spectrogram
%     tVec     — time axis (seconds)
%     fVec     — centre frequencies of Mel bands (Hz)

    NUM_BANDS   = 64;
    WIN_LENGTH  = round(fs * 0.025);   % 25 ms window
    HOP_LENGTH  = round(fs * 0.010);   % 10 ms hop
    FFT_LENGTH  = 2 ^ nextpow2(WIN_LENGTH);

    [melSpec, fVec, tVec] = melSpectrogram(audio, fs, ...
        "Window",          hann(WIN_LENGTH, "periodic"), ...
        "OverlapLength",   WIN_LENGTH - HOP_LENGTH, ...
        "FFTLength",       FFT_LENGTH, ...
        "NumBands",        NUM_BANDS, ...
        "FrequencyRange",  [0, fs/2]);

    % Convert to log scale (add small epsilon to avoid log(0))
    melSpec = 10 * log10(melSpec + 1e-6);
end


function display_results(melSpec, tVec, fVec, top_label, top_score, ...
                          all_scores, all_labels, source_name, model_name)
% DISPLAY_RESULTS  Plot the spectrogram and classification output.
%
%   all_labels  — string array of class names (musicnn), or [] (YAMNet)
%   model_name  — "yamnet" or "musicnn", used in the title

    figure("Name", "Music Classifier Results", "NumberTitle", "off", ...
           "Position", [100 100 900 650]);

    %% -- Mel Spectrogram --
    subplot(2, 1, 1);
    imagesc(tVec, fVec, melSpec);
    axis xy;
    colormap(hot);
    colorbar;
    xlabel("Time (s)");
    ylabel("Frequency (Hz)");
    title(sprintf("Mel Spectrogram — %s", source_name), ...
          "Interpreter", "none");

    %% -- Top Predictions Bar Chart --
    subplot(2, 1, 2);

    TOP_N = min(10, numel(all_scores));
    [sorted_scores, idx] = sort(all_scores, "descend");
    top_scores = sorted_scores(1:TOP_N) * 100;

    % Build tick labels: use named labels if available (musicnn),
    % otherwise fall back to class index numbers (YAMNet)
    if ~isempty(all_labels)
        tick_labels = all_labels(idx(1:TOP_N));
    else
        tick_labels = "Class " + string(idx(1:TOP_N));
    end

    % Colour bars: top prediction highlighted, rest muted
    colors = repmat([0.55 0.76 0.90], TOP_N, 1);
    colors(1, :) = [0.13 0.47 0.71];   % darker blue for winner

    barh(1:TOP_N, top_scores, "FaceColor", "flat", "CData", colors);
    set(gca, "YTick", 1:TOP_N, "YTickLabel", tick_labels, ...
             "YDir", "reverse");
    xlabel("Score (%)");
    title(sprintf("[%s]  Top: %s  (%.1f%%)", upper(model_name), ...
                  top_label, top_score * 100));
    xlim([0 100]);
    grid on;

    sgtitle("Music Genre / Instrument Classifier");
end
