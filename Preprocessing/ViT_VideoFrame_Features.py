import os
import numpy as np
import torch
from PIL import Image
import pickle
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTModel

# ====== PATHS (COLAB + DRIVE) ======
# Where all your HateMM-related pkls are stored
FOLDER_NAME = "/content/drive/MyDrive/HateMM/"

# Folder where this script will save per-video ViT features
VITF_FOLDER = os.path.join(FOLDER_NAME, "VITF")

# Folder where frameExtract.py wrote all the frames
# (you confirmed this earlier)
FRAMES_ROOT = "/content/drive/MyDrive/Dataset_Images"

os.makedirs(VITF_FOLDER, exist_ok=True)
print(f"Saving ViT features to: {VITF_FOLDER}")
print(f"Reading frames from: {FRAMES_ROOT}")

# ====== DEVICE SETUP ======
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using device:", device)

# ====== MODEL & FEATURE EXTRACTOR ======
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
model.to(device)
model.eval()

# ====== FRAME SAMPLING SETTINGS ======
minFrames = 100  # each video -> 100 frames
begin_frame, end_frame, skip_frame = 0, minFrames, 0  # (kept for compatibility)


def build_all_video_frame_lists(frames_root):
    """
    Scan FRAMES_ROOT and build:
        allVidList = [ [list of frame paths for video_1],
                       [list of frame paths for video_2],
                       ... ]
    Assumes structure:
        /content/Dataset_Images/hate_video_1/frame_0.jpg
        /content/Dataset_Images/hate_video_1/frame_1.jpg
        ...
        /content/Dataset_Images/non_hate_video_12/frame_0.jpg
        ...
    """
    allVidList = []

    if not os.path.exists(frames_root):
        raise FileNotFoundError(f"Frames root folder not found: {frames_root}")

    video_dirs = sorted(
        [
            d
            for d in os.listdir(frames_root)
            if os.path.isdir(os.path.join(frames_root, d))
        ]
    )

    print(f"Found {len(video_dirs)} video frame folders in {frames_root}")

    for vd in video_dirs:
        vdir = os.path.join(frames_root, vd)
        frame_files = sorted(
            [
                os.path.join(vdir, f)
                for f in os.listdir(vdir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        if len(frame_files) == 0:
            print(f"Warning: no frames found in {vdir}, skipping.")
            continue
        allVidList.append(frame_files)

    print(f"Total videos with frames: {len(allVidList)}")
    return allVidList


def read_images(frame_paths, min_frames=100):
    """
    Load frames for a single video and:
    - If frames <= min_frames: use all & pad with black images to min_frames
    - If frames > min_frames: sample uniformly to get exactly min_frames frames
    """
    X = []
    currFrameCount = 0
    videoFrameCount = len(frame_paths)

    if videoFrameCount <= min_frames:
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            X.append(image)
            currFrameCount += 1
            if currFrameCount == min_frames:
                break

        paddingImage = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB")
        while currFrameCount < min_frames:
            X.append(paddingImage)
            currFrameCount += 1
    else:
        step = int(videoFrameCount / min_frames)
        for i in range(0, videoFrameCount, step):
            image = Image.open(frame_paths[i]).convert("RGB")
            X.append(image)
            currFrameCount += 1
            if currFrameCount == min_frames:
                break

        paddingImage = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB")
        while currFrameCount < min_frames:
            X.append(paddingImage)
            currFrameCount += 1

    return X


def extract_vit_features(video_frames):
    """
    video_frames: list of full frame paths belonging to one video
    Saves a pickle: VITF/<video_id>_vit.pkl
    where video_id = folder name under Dataset_Images (e.g., 'hate_video_1')
    """
    # video_id = name of frame folder, i.e. parent directory of a frame
    video_id = os.path.basename(os.path.dirname(video_frames[0]))
    pickle_filename = os.path.join(VITF_FOLDER, f"{video_id}_vit.pkl")

    # Skip if already processed
    if os.path.exists(pickle_filename):
        # Uncomment to see which ones are skipped:
        # print(f"Already have VIT features for {video_id}, skipping.")
        return

    try:
        # Read and sample images
        video_images = read_images(video_frames, min_frames=minFrames)

        # Convert images to ViT inputs
        inputs = feature_extractor(images=video_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # outputs.last_hidden_state: [batch_size=100, seq_len, hidden_dim=768]
            last_hidden_states = outputs.last_hidden_state  # on device
            last_hidden_states = last_hidden_states.cpu()   # move to CPU

        # Take [CLS] token (index 0) for each frame
        # -> list of 100 vectors of shape (768,)
        video_features = [last_hidden_states[i, 0].numpy()
                          for i in range(minFrames)]

        with open(pickle_filename, "wb") as fp:
            pickle.dump(video_features, fp)

    except Exception as e:
        print(f"Error while processing video {video_id}: {e}")
        return


if __name__ == "__main__":
    # Build list of frames per video from Dataset_Images
    allVidList = build_all_video_frame_lists(FRAMES_ROOT)

    # Iterate and extract features
    for video_frames in tqdm(allVidList):
        extract_vit_features(video_frames)

    print("✅ ViT feature extraction completed.")
    print(f"Features saved in: {VITF_FOLDER}")
