import os
import sys
import shutil
import numpy as np
import pandas as pd

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional runtime dependency
    sf = None

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None

try:
    from Rmove_Noise import remove_background_noise
    from Baseline_Calibration import calibrate_and_normalize
    HAS_NOISE = True
except Exception as e:
    print(f"Noise reduction modules unavailable: {e}")
    remove_background_noise = None
    calibrate_and_normalize = None
    HAS_NOISE = False

try:
    from separate_speakers import separate_speakers
    HAS_DIARIZATION = True
except Exception as e:
    print(f"Diarization module unavailable: {e}")
    separate_speakers = None
    HAS_DIARIZATION = False

try:
    from Extract_Features import extract_egemaps_acoustic_features
    HAS_FEATURES = True
except Exception as e:
    print(f"Feature extraction module unavailable: {e}")
    extract_egemaps_acoustic_features = None
    HAS_FEATURES = False

# eGeMAPSv02 feature column names extracted via openSMILE
FEATURE_MAPPING = {
    "pitch": "F0semitoneFrom27.5Hz_sma3nz_mean",
    "pitch_var": "F0semitoneFrom27.5Hz_sma3nz_variance",
    "loudness": "Loudness_sma3_mean",
    "jitter": "jitterLocal_sma3nz_mean",
    "shimmer": "shimmerLocaldB_sma3nz_mean",
    "hnr": "HNRdBACF_sma3nz_mean",
    "flux": "spectralFlux_sma3_mean",
    "alpha": "alphaRatio_sma3_mean",
}

def clean_output_dir(directory):
    """Safely cleans and recreates the target output directory to prevent file mixing."""
    if os.path.exists(directory):
        print(f"Cleaning existing directory: '{directory}'...")
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def classify_state(z_scores):
    """
    Computes heuristic emotional/vocal state scores using standard Z-scores
    of eGeMAPS acoustic features, picking the highest matching state.
    """
    zp = z_scores.get("pitch", 0.0)
    zpv = z_scores.get("pitch_var", 0.0)
    zl = z_scores.get("loudness", 0.0)
    zji = z_scores.get("jitter", 0.0)
    zsh = z_scores.get("shimmer", 0.0)
    zhnr = z_scores.get("hnr", 0.0)
    zfl = z_scores.get("flux", 0.0)
    
    # Combined voice instability measure
    z_instability = (zji + zsh) / 2.0

    # 1. Stressed Score: High pitch, high loudness, high vocal fold jitter/shimmer, low HNR
    stressed = 0.35 * zp + 0.3 * zl + 0.25 * z_instability - 0.2 * zhnr
    
    # 2. Enthusiastic Score: High pitch, high pitch variability, high loudness, high spectral flux, good HNR
    enthusiastic = 0.25 * zp + 0.25 * zpv + 0.2 * zl + 0.2 * zfl + 0.1 * zhnr
    
    # 3. Hesitant Score: Low/hesitant loudness, low pitch, high instability (trembling/uncertainty), low flux
    hesitant = -0.4 * zl - 0.3 * zp + 0.3 * z_instability - 0.2 * zfl
    
    # 4. Confident Score: Moderate/steady loudness, moderate pitch, highly stable (low jitter/shimmer), high HNR
    confident = 0.25 * zl + 0.1 * zp - 0.25 * z_instability + 0.4 * zhnr

    scores = {
        "Stressed": stressed,
        "Enthusiastic": enthusiastic,
        "Hesitant": hesitant,
        "Confident": confident
    }
    
    # Return the state with the highest score
    return max(scores, key=scores.get), scores

def _run_builtin_fallback(input_audio, output_dir="separated_speakers_padded"):
    """
    Fallback pipeline when silero-vad / speechbrain are not installed.
    Analyses the raw audio directly in 10-second intervals without
    speaker diarisation and writes Classification_Report.md.
    """
    print("\n--- [Fallback] Running built-in audio analyser ---")
    if sf is None:
        print("soundfile is not installed – voice analysis skipped.")
        return

    try:
        audio, sample_rate = sf.read(input_audio, dtype="float32")
    except Exception as exc:
        print(f"Could not read audio file: {exc}")
        return

    if audio.ndim > 1:
        import numpy as _np
        audio = _np.mean(audio, axis=1)

    import numpy as np

    duration = len(audio) / sample_rate
    print(f"Audio duration: {duration:.1f}s @ {sample_rate}Hz")

    interval = 10  # seconds per chunk
    classified_records = []

    # -- Per-interval feature extraction & classification ---------------------
    # We do a single-pass Z-score by first collecting raw features for all
    # windows, then normalising across the recording.
    raw_windows = []
    for start_sec in range(0, int(duration), interval):
        end_sec   = min(start_sec + interval, duration)
        s_idx     = int(start_sec  * sample_rate)
        e_idx     = int(end_sec    * sample_rate)
        chunk     = audio[s_idx:e_idx]
        if len(chunk) == 0:
            continue

        rms      = float(np.sqrt(np.mean(chunk ** 2)))
        zcr      = float(np.mean(np.abs(np.diff(np.signbit(chunk))))) if len(chunk) > 1 else 0.0
        delta    = float(np.mean(np.abs(np.diff(chunk))))             if len(chunk) > 1 else 0.0
        jitter   = delta  / (rms + 1e-6)
        shimmer  = float(np.std(chunk)) / (rms + 1e-6) if rms > 1e-6 else 0.0
        hnr      = 1.0  / (1.0 + jitter + shimmer)
        flux     = delta  / (np.mean(chunk ** 2) + 1e-6)
        pitch    = 120.0 + zcr * 400.0
        pitch_v  = float(np.var(chunk))

        raw_windows.append({
            "interval_start": start_sec,
            "interval_end":   int(end_sec),
            "pitch":    pitch,
            "pitch_var": pitch_v,
            "loudness": rms,
            "jitter":   jitter,
            "shimmer":  shimmer,
            "hnr":      hnr,
            "flux":     flux,
            "alpha":    rms / (np.mean(np.abs(chunk)) + 1e-6),
        })

    if not raw_windows:
        print("No audio chunks found – voice analysis skipped.")
        return

    # Global stats for Z-score normalisation
    feat_keys = ["pitch", "pitch_var", "loudness", "jitter", "shimmer", "hnr", "flux", "alpha"]
    global_stats = {}
    for k in feat_keys:
        vals = np.array([w[k] for w in raw_windows])
        global_stats[k] = {"mean": float(np.mean(vals)),
                           "std":  max(float(np.std(vals)), 1e-6)}

    for w in raw_windows:
        z = {k: (w[k] - global_stats[k]["mean"]) / global_stats[k]["std"] for k in feat_keys}
        # Re-use classify_state with renamed keys
        z_mapped = {"pitch": z["pitch"], "pitch_var": z["pitch_var"],
                    "loudness": z["loudness"], "jitter": z["jitter"],
                    "shimmer": z["shimmer"], "hnr": z["hnr"],
                    "flux": z["flux"], "alpha": z["alpha"]}
        state, scores = classify_state(z_mapped)
        if w["loudness"] < 0.01:
            state = "Inactive"

        classified_records.append({
            "Speaker": "speaker001",
            "Time Interval": f"{w['interval_start']}s - {w['interval_end']}s",
            "Classified State": state,
            "Raw Features": {k: w[k] for k in feat_keys},
            "State Scores":  scores,
        })

    print(f"[Fallback] Classified {len(classified_records)} intervals.")
    generate_markdown_report("Classification_Report.md", classified_records, global_stats)
    print("Classification_Report.md saved.")


def run_voice_classification_pipeline(input_audio, output_dir="separated_speakers_padded", num_speakers=None):
    print("\n" + "="*50)
    print("STARTING UNIFIED VOICE CLASSIFIER PIPELINE")
    print("="*50)

    if not HAS_FEATURES:
        print("Falling back to the built-in audio analyzer because Extract_Features is unavailable.")
        return _run_builtin_fallback(input_audio, output_dir=output_dir)

    # Temporary filenames for intermediate processing
    noise_free_wav = "Noise_free.wav"
    calibrated_wav = "Calibrated_Audio.wav"
    
    current_audio = input_audio

    # Step 1: Remove Background Noise
    if HAS_NOISE:
        print("\n--- [Step 1/5] Removing Background Noise ---")
        remove_background_noise(input_audio, noise_free_wav, preserve_timeline=True)
        print("\n--- [Step 2/5] Calibrating & Normalizing Volume ---")
        calibrate_and_normalize(noise_free_wav, calibrated_wav, calibration_seconds=5.0)
        current_audio = calibrated_wav
    else:
        print("\n--- [Step 1/5 & 2/5] Noise Reduction Unavailable (Skipping) ---")

    # Step 3: Separate Speakers (Diarization)
    print("\n--- [Step 3/5] Separating Speakers ---")
    clean_output_dir(output_dir)
    
    use_diarization = HAS_DIARIZATION
    if use_diarization:
        try:
            separate_speakers(current_audio, output_dir=output_dir, num_speakers=num_speakers)
        except Exception as e:
            print(f"Diarization failed: {e}")
            use_diarization = False
            
    if not use_diarization:
        print("Diarization skipped/failed. Using single speaker fallback.")
        spk_dir = os.path.join(output_dir, "speaker001")
        os.makedirs(spk_dir, exist_ok=True)
        shutil.copy(current_audio, os.path.join(spk_dir, "combined_speech.wav"))

    # Get identified speaker folders
    speaker_folders = sorted([
        f for f in os.listdir(output_dir) 
        if f.startswith("speaker") and os.path.isdir(os.path.join(output_dir, f))
    ])

    if not speaker_folders:
        print("Error: No speakers separated. Pipeline aborted.")
        return

    print(f"\nFound {len(speaker_folders)} separated speaker(s). Running analysis...")

    # Step 4: Extract eGeMAPS Features per Speaker
    print("\n--- [Step 4/5] Extracting Acoustic Features ---")
    speaker_dfs = {}
    for spk in speaker_folders:
        spk_path = os.path.join(output_dir, spk)
        combined_wav = os.path.join(spk_path, "combined_speech.wav")
        csv_out = os.path.join(spk_path, "features.csv")
        
        if not os.path.exists(combined_wav):
            print(f"Warning: Combined track not found for {spk}. Skipping.")
            continue
            
        print(f"Extracting features for {spk} from '{combined_wav}'...")
        try:
            extract_egemaps_acoustic_features(combined_wav, csv_out, window_size=3.0, step_size=1.0)
            speaker_dfs[spk] = pd.read_csv(csv_out)
        except Exception as e:
            print(f"Error extracting features for {spk}: {e}")

    if not speaker_dfs:
        print("Error: No features extracted successfully. Pipeline aborted.")
        return

    # Step 5: Classify into Hesitant, Stressed, Enthusiastic, Confident in 10s Intervals
    print("\n--- [Step 5/5] Classifying Voice States in 10s Intervals ---")
    
    # 5.1 Chunk sliding-window data into 10-second segments for all speakers
    all_chunks = []
    
    for spk, df in speaker_dfs.items():
        if df.empty:
            continue
            
        # Group sliding frames into 10-second intervals based on window start time
        df['interval_10s'] = (df['window_start_sec'] // 10).astype(int) * 10
        
        # Calculate mean for each feature in the 10-second interval
        chunked = df.groupby('interval_10s').mean(numeric_only=True).reset_index()
        
        for _, row in chunked.iterrows():
            interval = int(row['interval_10s'])
            chunk_data = {
                "speaker": spk,
                "interval_start": interval,
                "interval_end": interval + 10,
            }
            # Copy over mapped eGeMAPS features
            for feat_key, feat_col in FEATURE_MAPPING.items():
                chunk_data[feat_key] = row.get(feat_col, 0.0)
            all_chunks.append(chunk_data)

    if not all_chunks:
        print("Error: No 10-second chunks could be assembled. Pipeline aborted.")
        return

    chunks_df = pd.DataFrame(all_chunks)

    # 5.2 Calculate Global Means and Standard Deviations across all chunks
    # This provides self-calibration relative to the recording session's properties
    global_stats = {}
    for feat_key in FEATURE_MAPPING.keys():
        values = chunks_df[feat_key].values
        global_stats[feat_key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)) if np.std(values) > 0 else 1.0
        }

    # 5.3 Classify each 10-second interval
    classified_records = []
    
    for idx, row in chunks_df.iterrows():
        # Compute local Z-scores for this chunk
        z_scores = {}
        for feat_key in FEATURE_MAPPING.keys():
            mean = global_stats[feat_key]["mean"]
            std = global_stats[feat_key]["std"]
            z_scores[feat_key] = (row[feat_key] - mean) / std

        # Run heuristic classification rules
        predicted_state, state_scores = classify_state(z_scores)
        
        # Safe thresholding for silence/inactive intervals
        # If absolute loudness is extremely low, mark as Silent/Inactive
        if row["loudness"] < 0.01:
            predicted_state = "Inactive"

        record = {
            "Speaker": row["speaker"],
            "Time Interval": f"{row['interval_start']}s - {row['interval_end']}s",
            "Classified State": predicted_state,
            "Raw Features": {f: float(row[f]) for f in FEATURE_MAPPING.keys()},
            "State Scores": {k: float(v) for k, v in state_scores.items()}
        }
        classified_records.append(record)

    # Output classification results as a DataFrame
    results_df = pd.DataFrame([
        {
            "Speaker": r["Speaker"],
            "Time Interval": r["Time Interval"],
            "Classified State": r["Classified State"]
        } for r in classified_records
    ])

    print("\n=== Pipeline Execution Success! ===")
    print(results_df.to_string(index=False))

    # Generate Markdown Report
    report_path = "Classification_Report.md"
    generate_markdown_report(report_path, classified_records, global_stats)
    print(f"\nDetailed classification report saved to: '{report_path}'")

def generate_markdown_report(report_path, records, global_stats):
    """Generates a styled, easy-to-read Markdown file containing classification timelines."""
    # Group records by speaker
    by_speaker = {}
    for r in records:
        spk = r["Speaker"]
        if spk not in by_speaker:
            by_speaker[spk] = []
        by_speaker[spk].append(r)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Voice Classification Timeline Report\n\n")
        f.write("This report documents the voice state classifications (Hesitant, Stressed, Enthusiastic, Confident) ")
        f.write("calculated in **10-second intervals** for each speaker.\n\n")
        
        f.write("## Heuristic Definitions & Calibration\n")
        f.write("The classifier calculates Z-scores relative to the average of this recording session to adjust ")
        f.write("for differences in speaker biology (gender, vocal pitch range) and microphone setups.\n\n")
        
        f.write("| State | Primary Acoustic Profile |\n")
        f.write("| :--- | :--- |\n")
        f.write("| **Confident** | High voice stability (low jitter/shimmer), strong resonance (high HNR), stable loudness. |\n")
        f.write("| **Enthusiastic** | High pitch, high pitch variability, elevated loudness, high spectral flux (dynamic). |\n")
        f.write("| **Stressed** | High pitch, elevated loudness, high vocal fold instability (jitter/shimmer), low HNR. |\n")
        f.write("| **Hesitant** | Low loudness, low pitch, high vocal fold jitter/shimmer (tremulous), flat articulation. |\n")
        f.write("| **Inactive** | Very low energy/volume levels (silence or non-speech duration). |\n\n")

        for spk, spk_records in sorted(by_speaker.items()):
            f.write(f"## 👤 {spk.capitalize()}\n\n")
            f.write("| Time Interval | Classified State | Dominant Characteristic / Feature Highlights |\n")
            f.write("| :--- | :--- | :--- |\n")
            
            for r in spk_records:
                state = r["Classified State"]
                raw = r["Raw Features"]
                scores = r["State Scores"]
                
                # Format a small highlight description based on feature states
                highlights = []
                if state == "Inactive":
                    highlights.append("Silent/No speech detected")
                else:
                    if raw["pitch"] > global_stats["pitch"]["mean"] + 0.5 * global_stats["pitch"]["std"]:
                        highlights.append("Elevated Pitch")
                    elif raw["pitch"] < global_stats["pitch"]["mean"] - 0.5 * global_stats["pitch"]["std"]:
                        highlights.append("Lower Pitch")
                        
                    if raw["jitter"] > global_stats["jitter"]["mean"]:
                        highlights.append("Vocal instability")
                    else:
                        highlights.append("Clear tone")
                        
                    if raw["loudness"] > global_stats["loudness"]["mean"]:
                        highlights.append("Loud projection")
                    else:
                        highlights.append("Soft projection")
                        
                highlights_str = ", ".join(highlights)
                
                f.write(f"| **{r['Time Interval']}** | **{state}** | {highlights_str} |\n")
            f.write("\n")

if __name__ == "__main__":
    # Target inputs
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "Video_Project.wav"

    if os.path.exists(input_file):
        num_speakers = None
        if len(sys.argv) > 2:
            try:
                num_speakers = int(sys.argv[2])
            except ValueError:
                num_speakers = None

        if num_speakers is not None:
            print(f"Using speaker count: {num_speakers}")
        else:
            print("Using auto speaker detection.")

        run_voice_classification_pipeline(
            input_file,
            num_speakers=num_speakers
        )

    else:
        print(f"Error: Target raw audio file '{input_file}' not found.")
        sys.exit(1)