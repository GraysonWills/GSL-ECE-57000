import os
import cv2
import pandas as pd
import glob
import queue
import threading
import signal
from concurrent.futures import ThreadPoolExecutor

# Global flag for graceful shutdown
shutdown_flag = threading.Event()

def signal_handler(signum, frame):
    print("Interrupt received, shutting down...")
    shutdown_flag.set()

signal.signal(signal.SIGINT, signal_handler)

def preprocess_frame(frame):
    return cv2.resize(frame, (256, 256))

def process_video(video_data, output_dir, q):
    video_id, raw_video_path = video_data
    sequence_output_dir = os.path.join(output_dir, video_id)
    
    if not os.path.exists(raw_video_path):
        return

    # Delete the contents of the previous output directory if it exists
    if os.path.exists(sequence_output_dir):
        shutil.rmtree(sequence_output_dir)

    os.makedirs(sequence_output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(raw_video_path, "*.png"))

    for image_file in sorted(image_files):
        if shutdown_flag.is_set():
            return

        frame = cv2.imread(image_file)
        processed_frame = preprocess_frame(frame)
        
        # Use the original filename
        original_filename = os.path.basename(image_file)
        frame_output_path = os.path.join(sequence_output_dir, original_filename)
        
        q.put((frame_output_path, processed_frame))

def writer_thread(q):
    while not shutdown_flag.is_set():
        try:
            frame_output_path, processed_frame = q.get(timeout=1)
            cv2.imwrite(frame_output_path, processed_frame)
            q.task_done()
        except queue.Empty:
            continue

def preprocess_dataset(dataset_csv, raw_videos_dir, output_dir):
    annotations = pd.read_csv(dataset_csv)
    #os.makedirs(output_dir, exist_ok=True)

    q = queue.Queue()
    writer = threading.Thread(target=writer_thread, args=(q,))
    writer.start()

    # Filter out videos that have already been processed
    video_data = []
    for _, row in annotations.iterrows():
        output_path = os.path.join(output_dir, row['id'])
        if not os.path.exists(output_path):
            video_data.append((row['id'], os.path.join(raw_videos_dir, row['id'], "1")))
        else:
            print(f"Skipping {row['id']} - already processed")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_video, vd, output_dir, q) for vd in video_data]

        for future in futures:
            future.result()

    q.join()
    shutdown_flag.set()
    writer.join()
if __name__ == '__main__':
    dataset_csv = "data/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/train.corpus.csv"
    raw_videos_dir = "data/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train"
    output_dir = "data/output_preprocess/"
    file_paths = [
        'data/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/train.corpus.csv',
        'data/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/test.corpus.csv',
        'data/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/dev.corpus.csv'
    ]
    
    # Process each existing file if it hasn't been split already
    if not os.path.exists(output_dir):
        # Create the output directory to mark as processed
     
    
        # Process each file
        for file_path in file_paths:
            if os.path.exists(file_path):
                # Load the data
                df = pd.read_csv(file_path, header=None, names=["data"])
    
                # Split the single column into multiple columns based on the '|' separator
                df = df["data"].str.split("|", expand=True)
    
                # Set the first row as the header
                df.columns = df.iloc[0]
                df = df[1:]  # Remove the first row since it's now the header
    
    
                # Overwrite the original file with the updated DataFrame without index numbers
                df.to_csv(file_path, index=False)

    preprocess_dataset(dataset_csv, raw_videos_dir, output_dir)

    # Count folders in output directory
    output_folders = len([name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name))])
    
    # Count folders in raw videos directory
    raw_folders = len([name for name in os.listdir(raw_videos_dir) if os.path.isdir(os.path.join(raw_videos_dir, name))])
    
    print(f"Number of folders in output directory: {output_folders}")
    print(f"Number of folders in raw videos directory: {raw_folders}")
