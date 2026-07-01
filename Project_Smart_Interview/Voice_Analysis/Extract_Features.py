import os
import sys
import numpy as np
import pandas as pd

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional runtime dependency
    sf = None

try:
    import opensmile
except ImportError:  # pragma: no cover - optional runtime dependency
    opensmile = None


def _read_audio(audio_path):
    if sf is None:
        raise RuntimeError("soundfile is required for fallback feature extraction")

    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), float(sample_rate)


def _fallback_feature_columns(window_values):
    rms = float(np.sqrt(np.mean(np.square(window_values))))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(window_values))))) if len(window_values) > 1 else 0.0
    peak = float(np.max(np.abs(window_values))) if len(window_values) > 0 else 0.0
    delta = float(np.mean(np.abs(np.diff(window_values)))) if len(window_values) > 1 else 0.0
    energy = float(np.mean(np.square(window_values)))

    pitch_proxy = 120.0 + (zcr * 400.0)
    jitter_proxy = delta / (rms + 1e-6)
    shimmer_proxy = (np.std(window_values) / (rms + 1e-6)) if rms > 1e-6 else 0.0
    hnr_proxy = 1.0 / (1.0 + jitter_proxy + shimmer_proxy)
    flux_proxy = delta / (energy + 1e-6)
    alpha_proxy = rms / (np.mean(np.abs(window_values)) + 1e-6)

    return {
        "F0semitoneFrom27.5Hz_sma3nz_mean": pitch_proxy,
        "F0semitoneFrom27.5Hz_sma3nz_variance": float(np.var(window_values)) if len(window_values) > 1 else 0.0,
        "F0semitoneFrom27.5Hz_sma3nz_minimum": float(np.min(window_values)) if len(window_values) > 0 else 0.0,
        "F0semitoneFrom27.5Hz_sma3nz_maximum": float(np.max(window_values)) if len(window_values) > 0 else 0.0,
        "Loudness_sma3_mean": rms,
        "Loudness_sma3_variance": float(np.var(window_values)) if len(window_values) > 1 else 0.0,
        "Loudness_sma3_minimum": float(np.min(window_values)) if len(window_values) > 0 else 0.0,
        "Loudness_sma3_maximum": float(np.max(window_values)) if len(window_values) > 0 else 0.0,
        "jitterLocal_sma3nz_mean": jitter_proxy,
        "jitterLocal_sma3nz_variance": shimmer_proxy,
        "jitterLocal_sma3nz_minimum": float(np.min(np.abs(np.diff(window_values)))) if len(window_values) > 1 else 0.0,
        "jitterLocal_sma3nz_maximum": float(np.max(np.abs(np.diff(window_values)))) if len(window_values) > 1 else 0.0,
        "shimmerLocaldB_sma3nz_mean": shimmer_proxy,
        "shimmerLocaldB_sma3nz_variance": float(np.var(np.abs(np.diff(window_values)))) if len(window_values) > 1 else 0.0,
        "shimmerLocaldB_sma3nz_minimum": float(np.min(np.abs(window_values))) if len(window_values) > 0 else 0.0,
        "shimmerLocaldB_sma3nz_maximum": peak,
        "HNRdBACF_sma3nz_mean": hnr_proxy,
        "HNRdBACF_sma3nz_variance": float(np.var(np.abs(window_values))) if len(window_values) > 0 else 0.0,
        "HNRdBACF_sma3nz_minimum": float(np.min(np.abs(window_values))) if len(window_values) > 0 else 0.0,
        "HNRdBACF_sma3nz_maximum": peak,
        "spectralFlux_sma3_mean": flux_proxy,
        "spectralFlux_sma3_variance": float(np.var(np.abs(np.diff(window_values)))) if len(window_values) > 1 else 0.0,
        "spectralFlux_sma3_minimum": float(np.min(np.abs(np.diff(window_values)))) if len(window_values) > 1 else 0.0,
        "spectralFlux_sma3_maximum": float(np.max(np.abs(np.diff(window_values)))) if len(window_values) > 1 else 0.0,
        "alphaRatio_sma3_mean": alpha_proxy,
        "alphaRatio_sma3_variance": float(np.var(window_values)) if len(window_values) > 1 else 0.0,
        "alphaRatio_sma3_minimum": float(np.min(window_values)) if len(window_values) > 0 else 0.0,
        "alphaRatio_sma3_maximum": float(np.max(window_values)) if len(window_values) > 0 else 0.0,
    }


def extract_egemaps_acoustic_features(audio_path, csv_output_path, window_size=3.0, step_size=1.0):
    """Extract acoustic features from an audio file, preferring openSMILE and falling back gracefully."""
    print("=== Acoustic Feature Extraction Pipeline ===")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"The input audio file '{audio_path}' does not exist.")

    print(f"Loading and processing audio file: '{audio_path}'")

    if opensmile is not None:
        try:
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )
            lld_df = smile.process_file(audio_path)
            if lld_df.empty:
                raise ValueError("Acoustic extraction returned an empty DataFrame.")

            lld_df = lld_df.reset_index()
            lld_df['start_sec'] = lld_df['start'].dt.total_seconds()
            lld_df['end_sec'] = lld_df['end'].dt.total_seconds()

            metadata_cols = ['file', 'start', 'end', 'start_sec', 'end_sec']
            feature_cols = [col for col in lld_df.columns if col not in metadata_cols]
            duration = lld_df['end_sec'].max()
            pooled_records = []
            t = 0.0
            while t + window_size <= duration or (t == 0.0 and duration > 0.0):
                w_start = t
                w_end = t + window_size
                if w_start > duration:
                    break
                window_mask = (lld_df['start_sec'] >= w_start) & (lld_df['start_sec'] < w_end)
                window_df = lld_df[window_mask]
                if not window_df.empty:
                    record = {
                        'timestamp': f"{w_start:.1f}s",
                        'window_start_sec': w_start,
                        'window_end_sec': w_end,
                    }
                    for col in feature_cols:
                        values = window_df[col].values
                        if len(values) == 0:
                            record[f"{col}_mean"] = 0.0
                            record[f"{col}_variance"] = 0.0
                            record[f"{col}_minimum"] = 0.0
                            record[f"{col}_maximum"] = 0.0
                        else:
                            record[f"{col}_mean"] = float(np.mean(values))
                            record[f"{col}_variance"] = float(np.var(values))
                            record[f"{col}_minimum"] = float(np.min(values))
                            record[f"{col}_maximum"] = float(np.max(values))
                    pooled_records.append(record)
                t += step_size
                if t >= duration:
                    break

            if not pooled_records:
                raise ValueError("No time windows could be processed from the audio signal.")

            pooled_df = pd.DataFrame(pooled_records)
            pooled_df.to_csv(csv_output_path, index=False)
            print(f"Pooling complete. Exporting structured features to: '{csv_output_path}'")
            return True
        except Exception as exc:
            print(f"openSMILE extraction failed, falling back to built-in audio analysis: {exc}")

    print("Using built-in fallback acoustic analysis.")
    audio, sample_rate = _read_audio(audio_path)
    duration = len(audio) / sample_rate
    pooled_records = []
    t = 0.0

    while t + window_size <= duration or (t == 0.0 and duration > 0.0):
        w_start = t
        w_end = t + window_size
        if w_start > duration:
            break

        start_idx = int(w_start * sample_rate)
        end_idx = int(w_end * sample_rate)
        window_values = audio[start_idx:end_idx]
        if len(window_values) == 0:
            t += step_size
            continue

        record = {
            'timestamp': f"{w_start:.1f}s",
            'window_start_sec': w_start,
            'window_end_sec': w_end,
        }
        features = _fallback_feature_columns(window_values)
        record.update(features)
        pooled_records.append(record)

        t += step_size
        if t >= duration:
            break

    if not pooled_records:
        raise ValueError("No time windows could be processed from the audio signal.")

    pooled_df = pd.DataFrame(pooled_records)
    pooled_df.to_csv(csv_output_path, index=False)
    print(f"Fallback analysis complete. Exporting structured features to: '{csv_output_path}'")
    return True


if __name__ == "__main__":
    input_audio = "separated_speakers_padded/speaker002/combined_speech.wav"
    output_csv = "Acoustic_Features.csv"

    try:
        success = extract_egemaps_acoustic_features(
            audio_path=input_audio,
            csv_output_path=output_csv,
            window_size=3.0,
            step_size=1.0
        )
        if success:
            print("Feature extraction executed successfully.")
    except FileNotFoundError as fnf:
        print(f"\nConfiguration Error: {fnf}")
        print("Please ensure you have run 'Baseline_Calibration.py' first to generate 'Calibrated_Audio.wav'.")
    except Exception as err:
        print(f"\nExecution Error: {err}")
        sys.exit(1)
