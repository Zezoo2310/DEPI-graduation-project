import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, collect_chunks, save_audio

def remove_background_noise(audio_path, output_path, sampling_rate=16000, preserve_timeline=True):
    model = load_silero_vad()
    wav = read_audio(audio_path, sampling_rate=sampling_rate)

    speech_timestamps = get_speech_timestamps(
        wav, 
        model, 
        sampling_rate=sampling_rate,
        threshold=0.5,             # 0.5 is standard; higher means more aggressive noise cutting
        min_speech_duration_ms=250, # ignore clicks or short random noises under 250ms
        min_silence_duration_ms=400 # keep breathing/natural pauses together if under 400ms
    )
    if speech_timestamps:
        if preserve_timeline:
            print(f"Detected {len(speech_timestamps)} active speech segments. Preserving timeline (muting noise/silence)...")
            clean_wav = torch.zeros_like(wav)
            for segment in speech_timestamps:
                start = segment['start']
                end = segment['end']
                clean_wav[start:end] = wav[start:end]
            save_audio(output_path, clean_wav, sampling_rate=sampling_rate)
        else:
            print(f"Detected {len(speech_timestamps)} active speech segments. Gluing chunks...")
            clean_audio_chunks = collect_chunks(speech_timestamps, wav)
            save_audio(output_path, clean_audio_chunks, sampling_rate=sampling_rate)
            
        print(f"Success! Noise-free audio saved to: {output_path}") 
    else:
        print("No human speech was detected in the audio file.")

if __name__ == "__main__":
    remove_background_noise("Video Project.mp3", "Noise_free.wav")