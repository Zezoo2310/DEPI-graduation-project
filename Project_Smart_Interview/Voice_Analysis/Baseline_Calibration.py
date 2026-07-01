import os
import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, save_audio

def compute_rms(tensor):
    if len(tensor) == 0:
        return 0.0
    return torch.sqrt(torch.mean(tensor ** 2)).item()

def calibrate_and_normalize(audio_path, output_path, calibration_seconds=5.0, sampling_rate=16000):
    print("=== [1/4] Loading VAD Model & Audio ===")
    model = load_silero_vad()
    wav = read_audio(audio_path, sampling_rate=sampling_rate)
    duration_sec = len(wav) / sampling_rate
    print(f"Audio loaded. Duration: {duration_sec:.2f} seconds.")

    print("\n=== [2/4] Detecting Speech Segments ===")
    speech_timestamps = get_speech_timestamps(
        wav, 
        model, 
        sampling_rate=sampling_rate,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=400
    )
    
    num_segments = len(speech_timestamps)
    print(f"Detected {num_segments} speech segments.")
    if num_segments == 0:
        print("No speech detected. Saving silent audio file.")
        save_audio(output_path, torch.zeros_like(wav), sampling_rate=sampling_rate)
        return

    print("\n=== [3/4] Calibrating Volume Baseline ===")
    calibration_samples = int(calibration_seconds * sampling_rate)
    calibration_chunks = []
    
    for segment in speech_timestamps:
        start = segment['start']
        end = segment['end']
        # Collect chunks that start within the calibration window
        if start < calibration_samples:
            chunk_end = min(end, calibration_samples)
            if chunk_end > start:
                calibration_chunks.append(wav[start:chunk_end])

    if calibration_chunks:
        calibration_tensor = torch.cat(calibration_chunks)
        baseline_rms = compute_rms(calibration_tensor)
        print(f"Calculated baseline RMS of {baseline_rms:.5f} from the first {calibration_seconds} seconds of speech.")
    else:
        # Fallback to the first detected segment
        first_seg = speech_timestamps[0]
        baseline_rms = compute_rms(wav[first_seg['start']:first_seg['end']])
        print(f"No speech in first {calibration_seconds}s. Using first speech segment RMS as baseline: {baseline_rms:.5f}")
    
    # If the baseline RMS is extremely low (e.g. silence), fallback to a standard target
    if baseline_rms < 1e-4:
        baseline_rms = 0.05
        print(f"Baseline RMS too low. Using default target RMS: {baseline_rms}")

    print("\n=== [4/4] Normalizing Speech Segments (AGC) ===")
    clean_wav = torch.zeros_like(wav)
    
    for idx, segment in enumerate(speech_timestamps):
        start = segment['start']
        end = segment['end']
        segment_wav = wav[start:end]
        seg_rms = compute_rms(segment_wav)
        
        if seg_rms > 1e-6:
            # Compute required gain to match baseline
            gain = baseline_rms / seg_rms
            
            # Constrain gain to a safe range to avoid amplifying silence/noise excessively
            gain = max(0.2, min(gain, 5.0))
            
            normalized_segment = segment_wav * gain
            
            # Apply peak limiting to prevent clipping
            peak = torch.max(torch.abs(normalized_segment)).item()
            if peak > 0.95:
                normalized_segment = normalized_segment * (0.95 / peak)
                
            new_rms = compute_rms(normalized_segment)
            print(f"Segment {idx+1:02d} ({start/sampling_rate:6.2f}s -> {end/sampling_rate:6.2f}s): "
                  f"Orig RMS: {seg_rms:.4f} | Gain: {gain:4.2f}x | New RMS: {new_rms:.4f}")
            
            clean_wav[start:end] = normalized_segment
        else:
            clean_wav[start:end] = segment_wav

    save_audio(output_path, clean_wav, sampling_rate=sampling_rate)
    print(f"\nSuccess! Calibrated and normalized audio saved to: '{output_path}'")

if __name__ == "__main__":
    # Calibrate on the original audio project, saving to Calibrated_Audio.wav
    input_file = "Video Project.mp3"
    output_file = "Calibrated_Audio.wav"
    
    if os.path.exists(input_file):
        calibrate_and_normalize(input_file, output_file, calibration_seconds=5.0)
    else:
        print(f"Error: Input file '{input_file}' not found.")
