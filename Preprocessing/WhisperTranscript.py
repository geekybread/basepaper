import os
import soundfile as sf
import librosa
import torch
from transformers import pipeline
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Folder containing NON-HATE videos
video_folder = "/content/drive/MyDrive/Dataset/non_hate_videos"

#for hate videos, uncomment the line below
#video_folder = "/content/drive/MyDrive/Dataset/hate_videos"

# Save transcripts inside the same folder
# E.g. non_hate_video_1_whisper_tiny.txt
processed_log_path = "/content/drive/MyDrive/HateMM/processed_audios_nonhate.txt"

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30,
    device=0 if torch.cuda.is_available() else -1,
)

def load_processed_audios():
    if os.path.exists(processed_log_path):
        with open(processed_log_path, 'r') as file:
            return set(file.read().splitlines())
    return set()

def log_processed_audio(video_name):
    with open(processed_log_path, 'a') as file:
        file.write(video_name + "\n")

def process_audios(video_folder):
    processed = load_processed_audios()

    for video_file in tqdm(os.listdir(video_folder)):
        if not video_file.endswith(".mp4"):
            continue

        if video_file in processed:
            continue  # skip already processed

        video_path = os.path.join(video_folder, video_file)
        transcript_path = video_path.replace(".mp4", "_whisper_tiny.txt")

        print(f"Processing {video_file}...")

        try:
            # load audio using librosa
            audio, sr = librosa.load(video_path, sr=16000, mono=True)

            transcript = pipe(audio)["text"]

            with open(transcript_path, "w") as f:
                f.write(transcript)

            log_processed_audio(video_file)

        except Exception as e:
            print(f"Failed on {video_file}: {e}")

    print("Transcription completed for non-hate videos.")

if __name__ == "__main__":
    process_audios(video_folder)
