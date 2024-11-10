import random
import cv2

def augment_frame(frame):
    # Random horizontal flip
    if random.random() > 0.5:
        frame = cv2.flip(frame, 1)  # Horizontal flip
    return frame
