% =========================================================
% main.m  —  Music Genre / Instrument Classifier
% Supports live microphone input and pre-recorded audio files.
% Supports two classifier backends: YAMNet and musicnn.
%
% Requirements:
%   - Python 3.9+ with dependencies installed (see README)
%   - MATLAB Audio Toolbox
%   - pyenv configured (run `pyenv("Version","3.x")` if needed)
%
% Usage:
%   main()                        % prompts for source + model
%   main("file")                  % load from file picker, prompts model
%   main("mic")                   % record from mic, prompts model
%   main("file", "yamnet")        % file + YAMNet (521 AudioSet classes)
%   main("file", "musicnn")       % file + musicnn (50 music-specific tags)
%   main("mic",  "musicnn")       % mic  + musicnn
% =========================================================

function main(source, model)

    %% --- 0. Argument defaults ---
    if nargin < 1
        src_choice = menu("Audio Source", "Load File", "Record from Mic");
        source = ["file", "mic"](src_choice);
    end
    if nargin < 2
        mdl_choice = menu("Classifier", "YAMNet (broad AudioSet)", ...
                                        "musicnn (music-specific)");
        model = ["yamnet", "musicnn"](mdl_choice);
    end

    %% --- 1. Acquire Audio ---
    switch source
        case "file"
            [file, path] = uigetfile({"*.mp3;*.wav;*.flac;*.ogg", "Audio Files"});
            if isequal(file, 0), disp("Cancelled."); return; end
            [audio, fs] = audioread(fullfile(path, file));
            label_source = file;

        case "mic"
            fs       = 16000;
            duration = 5;
            fprintf("Recording %d seconds from microphone...\n", duration);
            recorder = audioDeviceReader("SampleRate", fs, ...
                                         "SamplesPerFrame", fs * duration);
            setup(recorder);
            audio    = recorder();
            release(recorder);
            disp("Recording complete.");
            label_source = "microphone";

        otherwise
            error("source must be ''file'' or ''mic''");
    end

    %% --- 2. Preprocess ---
    audio = preprocess_audio(audio, fs);

    %% --- 3. Extract Features (for visualisation) ---
    [melSpec, tVec, fVec] = extract_features(audio, fs);

    %% --- 4. Classify via Python ---
    fprintf("Running %s classifier...\n", model);

    switch model

        % ── YAMNet ──────────────────────────────────────────────────────
        case "yamnet"
            result = pyrunfile("yamnet_wrapper.py", ...
                               ["top_label", "top_score", "all_scores"], ...
                               audio_data  = py.numpy.array(audio), ...
                               sample_rate = int32(fs));

            top_label  = string(result{1});
            top_score  = double(result{2});
            all_scores = double(py.array.array("d", result{3}));
            all_labels = [];   % YAMNet labels loaded inside Python

        % ── musicnn ─────────────────────────────────────────────────────
        case "musicnn"
            % musicnn needs a file path → write a temp wav
            tmp_path = fullfile(tempdir, "classifier_tmp.wav");
            audiowrite(tmp_path, audio, 16000);

            result = pyrunfile("musicnn_wrapper.py", ...
                               ["top_label", "top_score", ...
                                "all_scores", "all_labels"], ...
                               audio_path = py.str(tmp_path), ...
                               topn       = int32(10));

            top_label  = string(result{1});
            top_score  = double(result{2});
            all_scores = double(py.array.array("d", result{3}));
            all_labels = cellfun(@string, cell(result{4}));

            delete(tmp_path);   % clean up temp file

        otherwise
            error("model must be ''yamnet'' or ''musicnn''");
    end

    %% --- 5. Display Results ---
    display_results(melSpec, tVec, fVec, top_label, top_score, ...
                    all_scores, all_labels, label_source, model);
end
