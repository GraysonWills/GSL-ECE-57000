import cv2
import numpy as np

def preprocess_frame(frame):
    # Resize frames to a consistent size of 256x256 and crop to 224x224
    resized_frame = cv2.resize(frame, (256, 256))
    cropped_frame = resized_frame[16:240, 16:240]
    return cropped_frame

def normalize_frame(frame, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # Normalize pixel values based on dataset mean and standard deviation
    frame = frame / 255.0
    frame = (frame - mean) / std
    return frame
