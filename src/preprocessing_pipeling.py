import os
import cv2
from src.utils.data_preprocessing import preprocess_frame
import pandas as pd

def preprocess_dataset(dataset_csv, raw_videos_dir, output_dir):
    # Load dataset annotations
    annotations = pd.read_csv(dataset_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, row in annotations.iterrows():
        video_path = os.path.join(raw_videos_dir, row['video_filename'])
        output_path = os.path.join(output_dir, row['video_filename'])
        
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Preprocess each frame using preprocess_frame
            frames.append(preprocess_frame(frame))

        cap.release()

        # Save the processed frames as a new video
        save_path = os.path.join(output_dir, row['video_filename'])
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()

if __name__ == '__main__':
    # Paths for input data and output
    dataset_csv = "../data/PHOENIX14/splits/train_annotations.csv"  # Annotations CSV file
    raw_videos_dir = "../data/PHOENIX14/raw_videos/"                # Directory containing raw videos
    output_dir = "../data/PHOENIX14/processed_videos/"              # Directory to store processed videos

    # Run preprocessing
    preprocess_dataset(dataset_csv, raw_videos_dir, output_dir)
