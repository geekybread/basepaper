import os
import cv2
from tqdm import tqdm

# Where to save all extracted frames (shared by all videos)
FRAMES_ROOT = "/content/Dataset_Images"

# Where your videos live
VIDEO_FOLDERS = [
    ("/content/drive/MyDrive/Dataset/hate_videos", "hate"),
    ("/content/drive/MyDrive/Dataset/non_hate_videos", "non_hate"),
]

os.makedirs(FRAMES_ROOT, exist_ok=True)


def extract_frames(video_path, target_folder):
    # Try to open video first
    cap = cv2.VideoCapture(video_path)
    success, _ = cap.read()
    if not success:
        print(f"Failed to read video: {video_path}")
        cap.release()
        return

    # Create folder for this video's frames
    os.makedirs(target_folder, exist_ok=True)

    # If frames already exist, skip
    if os.listdir(target_folder):
        # print(f"Frames already extracted for video: {video_path}")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    count = 0

    while True:
        # Set position in milliseconds (1 frame per second)
        cap.set(cv2.CAP_PROP_POS_MSEC, count * 1000)
        success, img = cap.read()
        if not success:
            break

        frame_path = os.path.join(target_folder, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, img)
        count += 1

    cap.release()


if __name__ == "__main__":
    for folder_path, label in VIDEO_FOLDERS:
        if not os.path.isdir(folder_path):
            print(f"WARNING: folder does not exist: {folder_path}")
            continue

        print(f"Processing {label} videos from: {folder_path}")
        video_files = sorted(os.listdir(folder_path))

        for f in tqdm(video_files, desc=f"{label}_videos"):
            if not f.lower().endswith(".mp4"):
                continue

            video_path = os.path.join(folder_path, f)
            video_id = os.path.splitext(f)[0]  # e.g., "hate_video_1"
            target_folder = os.path.join(FRAMES_ROOT, video_id)

            extract_frames(video_path, target_folder)

    print("✅ Frame extraction completed for all available videos.")
    print(f"Frames saved under: {FRAMES_ROOT}")
