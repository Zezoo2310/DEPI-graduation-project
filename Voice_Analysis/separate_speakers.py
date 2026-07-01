import os
import sys
import torch
import numpy as np
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, save_audio
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering

def separate_speakers(audio_path, output_dir='separated_speakers', num_speakers=None, distance_threshold=0.5, sampling_rate=16000):
    """
    Separates speakers in the given audio file and saves each speaker's clips and combined audio into folders.
    """
    # 1. Load Models
    print("=== [1/5] Loading Models ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    print("Loading Silero VAD model...")
    vad_model = load_silero_vad()
    
    print("Loading SpeechBrain ECAPA-TDNN speaker embedding model...")
    # EncoderClassifier automatically caches files locally
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        run_opts={"device": device}
    )
    
    # 2. Load Audio
    print("\n=== [2/5] Loading Audio ===")
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        sys.exit(1)
        
    print(f"Reading audio from '{audio_path}'...")
    wav = read_audio(audio_path, sampling_rate=sampling_rate)
    duration_sec = len(wav) / sampling_rate
    print(f"Audio loaded. Duration: {duration_sec:.2f} seconds.")
    
    # 3. Detect Speech Segments
    print("\n=== [3/5] Detecting Speech Segments ===")
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sampling_rate,
        return_seconds=True,
        min_speech_duration_ms=400,  # Focus on segments containing enough vocal info
        min_silence_duration_ms=100
    )
    
    num_segments = len(speech_timestamps)
    print(f"Detected {num_segments} speech segments.")
    
    if num_segments == 0:
        print("No speech detected. Exiting.")
        return

    # 4. Extract Speaker Embeddings
    print("\n=== [4/5] Extracting Speaker Embeddings ===")
    embeddings = []
    valid_segments = []
    
    for i, segment in enumerate(speech_timestamps):
        start_sec = segment['start']
        end_sec = segment['end']
        
        # Discard segments shorter than 0.5s as they don't carry enough speaker info
        if (end_sec - start_sec) < 0.5:
            continue
            
        start_sample = int(start_sec * sampling_rate)
        end_sample = int(end_sec * sampling_rate)
        
        # Slice waveform
        chunk = wav[start_sample:end_sample]
        
        # Extract embedding using SpeechBrain
        with torch.no_grad():
            # encode_batch expects (batch, time) shape
            embedding_tensor = classifier.encode_batch(chunk.unsqueeze(0))
            # Embedding shape is [1, 1, 192], we squeeze it to a 1D vector of 192 elements
            emb = embedding_tensor.squeeze().cpu().numpy()
            
        embeddings.append(emb)
        valid_segments.append(segment)
        
        if (i + 1) % 10 == 0 or (i + 1) == num_segments:
            print(f"Processed {i + 1}/{num_segments} segments...")
            
    print(f"Extracted speaker embeddings for {len(valid_segments)} valid segments.")
    
    if len(valid_segments) == 0:
        print("No speech segments were long enough to extract speaker features. Exiting.")
        return

    # 5. Cluster Speakers
    print("\n=== [5/5] Clustering Speakers ===")
    embeddings_arr = np.array(embeddings)
    
    # Normalize embeddings to unit length (important for Cosine similarity)
    norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
    embeddings_arr = embeddings_arr / np.maximum(norms, 1e-10)
    
    if len(valid_segments) == 1:
        print("Only one segment detected. Assigning to Speaker 1.")
        speaker_labels = [0]
    else:
        # Determine clustering approach
        if num_speakers is not None:
            print(f"Clustering into exactly {num_speakers} speakers...")
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers, 
                metric='cosine', 
                linkage='average'
            )
        else:
            print(f"Auto-detecting speakers using distance threshold: {distance_threshold}...")
            clustering = AgglomerativeClustering(
                n_clusters=None, 
                distance_threshold=distance_threshold, 
                metric='cosine', 
                linkage='average'
            )
            
        speaker_labels = clustering.fit_predict(embeddings_arr)
        
    num_detected_speakers = len(set(speaker_labels))
    print(f"Successfully identified {num_detected_speakers} speaker(s).")
    
    # 6. Group and Save Results
    print(f"\nSaving speaker clips into directory: '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Group segments and audio tensors by speaker
    speaker_chunks = {spk_id: [] for spk_id in range(num_detected_speakers)}
    speaker_segment_counts = {spk_id: 0 for spk_id in range(num_detected_speakers)}
    
    for idx, segment in enumerate(valid_segments):
        spk_id = speaker_labels[idx]
        start_sec = segment['start']
        end_sec = segment['end']
        
        start_sample = int(start_sec * sampling_rate)
        end_sample = int(end_sec * sampling_rate)
        chunk_tensor = wav[start_sample:end_sample]
        
        # Create speaker folder path (e.g., extracted_speakers/speaker001)
        spk_folder_name = f"speaker{spk_id + 1:03d}"
        spk_dir = os.path.join(output_dir, spk_folder_name)
        os.makedirs(spk_dir, exist_ok=True)
        
        # Save individual clip
        clip_index = speaker_segment_counts[spk_id] + 1
        clip_filename = f"clip_{clip_index:03d}_{start_sec:.1f}s_to_{end_sec:.1f}s.wav"
        clip_filepath = os.path.join(spk_dir, clip_filename)
        save_audio(clip_filepath, chunk_tensor, sampling_rate=sampling_rate)
        
        # Track for combined file
        speaker_chunks[spk_id].append(chunk_tensor)
        speaker_segment_counts[spk_id] += 1

    # Save a combined/glued track for each speaker
    print("\nGenerating combined continuous tracks for each speaker...")
    for spk_id, chunks in speaker_chunks.items():
        if not chunks:
            continue
            
        spk_folder_name = f"speaker{spk_id + 1:03d}"
        spk_dir = os.path.join(output_dir, spk_folder_name)
        
        # Glue the PyTorch tensors together
        combined_tensor = torch.cat(chunks)
        combined_filepath = os.path.join(spk_dir, "combined_speech.wav")
        save_audio(combined_filepath, combined_tensor, sampling_rate=sampling_rate)
        
        print(f" - {spk_folder_name}: Saved {len(chunks)} clips & 'combined_speech.wav' ({len(combined_tensor)/sampling_rate:.2f}s)")
        
    print(f"\nAll files saved successfully inside: '{output_dir}/'")

if __name__ == "__main__":
    # You can change 'Video Project.mp3' to 'Noise_free.wav' to run on the noise-free audio!
    input_audio = "Calibrated_Audio.wav"
    
    # Set to a number (e.g., 2) if you want to force exactly 2 speakers, or None to auto-detect
    num_speakers = 5
    
    separate_speakers(
        audio_path=input_audio, 
        output_dir="separated_speakers",
        num_speakers=num_speakers,
        distance_threshold=0.5
    )
