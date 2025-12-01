import os
import traceback
import pickle
import librosa
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoProcessor, ClapAudioModel
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# Where to save feature .pkl files
FOLDER_NAME = "/content/drive/MyDrive/HateMM/"

# Where your videos live
HATE_VIDEOS_DIR = "/content/drive/MyDrive/Dataset/hate_videos"
NON_HATE_VIDEOS_DIR = "/content/drive/MyDrive/Dataset/non_hate_videos"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_all_video_paths():
    """Collect all .mp4 files from hate and non-hate folders."""
    video_paths = []

    for folder in [HATE_VIDEOS_DIR, NON_HATE_VIDEOS_DIR]:
        if not os.path.isdir(folder):
            print(f"WARNING: Folder not found: {folder}")
            continue

        for fname in os.listdir(folder):
            if fname.lower().endswith(".mp4"):
                video_paths.append(os.path.join(folder, fname))

    print(f"Found {len(video_paths)} video files in total.")
    return video_paths


def extract_features(audio_path, feature_type):
    """
    Extract CLAP or Wav2Vec2 audio features from a single video file.
    """
    if feature_type == "CLAP":
        # Load CLAP model + processor (audio encoder only)
        processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
        model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused").to(device)

        # Multi-GPU (not really relevant in Colab, but harmless)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs for CLAP!")
            model = nn.DataParallel(model)

        # Load audio from video (mono, 48k for CLAP)
        audio, _ = librosa.load(audio_path, sr=48000, mono=True)

        # Process waveform
        inputs = processor(
            audios=audio,
            return_tensors="pt",
            sampling_rate=48000
        )

        # Move tensors to correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state  # [1, T, D]

        features = features.cpu().numpy()

    elif feature_type == "Wav2Vec2":
        # Load Wav2Vec2 model + feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h"
        ).to(device)

        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs for Wav2Vec2!")
            model = nn.DataParallel(model)

        # Load audio from video (mono, 16k for Wav2Vec2)
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)

        # Chunk long audio into 30s segments
        chunk_size = 30 * 16000
        audio_chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

        all_features = []
        for chunk in audio_chunks:
            if len(chunk) == 0:
                continue
            inputs = feature_extractor(
                chunk,
                return_tensors="pt",
                sampling_rate=16000
            ).input_values.to(device)

            with torch.no_grad():
                outputs = model(input_values=inputs)
                feats = outputs.last_hidden_state  # [1, T, D]
                all_features.append(feats.cpu().numpy().squeeze(0))  # [T, D]

        if len(all_features) == 0:
            print(f"No valid audio chunks for {audio_path}")
            return None

        # Concatenate over time dimension
        features = np.concatenate(all_features, axis=0)  # [T_total, D]

    else:
        print("Invalid feature type:", feature_type)
        return None

    return features


def extract_all_features(feature_type):
    """
    Loop over all videos and extract audio features of the given type.
    Returns:
        allAudioFeatures: dict { 'hate_video_1.mp4': feature_array, ... }
        failedList: list of paths that failed
    """
    video_paths = get_all_video_paths()
    allAudioFeatures = {}
    failedList = []

    for video_path in tqdm(video_paths):
        video_name = os.path.basename(video_path)
        if video_name in allAudioFeatures:
            # In case of duplicates somehow
            continue

        try:
            feats = extract_features(video_path, feature_type)
            if feats is None:
                print(f"Got None features for {video_path}, skipping.")
                failedList.append(video_path)
                continue

            # Use the MP4 filename as key, consistent with other .pkl files
            allAudioFeatures[video_name] = feats

        except Exception as e:
            failedList.append(video_path)
            print(f"\nFailed to extract {feature_type} features for {video_path}")
            print(f"Error: {e}")
            traceback.print_exc()

    return allAudioFeatures, failedList


if __name__ == "__main__":
    os.makedirs(FOLDER_NAME, exist_ok=True)

    # --------- CLAP FEATURES ----------
    print("\n=== Extracting CLAP audio features ===")
    clap_features, clap_failed = extract_all_features("CLAP")
    print(f"\nCLAP: successfully processed {len(clap_features)} videos.")
    print(f"CLAP: failed videos count = {len(clap_failed)}")

    clap_out_path = os.path.join(FOLDER_NAME, "CLAP_features.pkl")
    with open(clap_out_path, "wb") as fp:
        pickle.dump(clap_features, fp)
    print(f"Saved CLAP features to: {clap_out_path}")

    # --------- WAV2VEC2 FEATURES (OPTIONAL - COMMENTED BY DEFAULT) ----------
    # This is heavier. Uncomment when/if you want these too.

    # print("\n=== Extracting Wav2Vec2 audio features ===")
    # w2v_features, w2v_failed = extract_all_features("Wav2Vec2")
    # print(f"\nWav2Vec2: successfully processed {len(w2v_features)} videos.")
    # print(f"Wav2Vec2: failed videos count = {len(w2v_failed)}")
    #
    # w2v_out_path = os.path.join(FOLDER_NAME, "Wav2Vec2_features_chunked.pkl")
    # with open(w2v_out_path, "wb") as fp:
    #     pickle.dump(w2v_features, fp)
    # print(f"Saved Wav2Vec2 features to: {w2v_out_path}")
