import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image
import random

class PhoenixDataset(Dataset):
    def __init__(self, annotations, frames_dir, mode='train'):
        self.annotations = annotations
        self.frames_dir = frames_dir
        self.mode = mode

        # Define transformations for training and inference
        self.transform = self.get_transforms()

        # Filter annotations to only include existing folders
        self.annotations = self.annotations[self.annotations['id'].apply(
            lambda x: os.path.exists(os.path.join(self.frames_dir, x))
        )]

    def get_transforms(self):
        if self.mode == 'train':
            return transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_id = self.annotations.iloc[idx]['id']
        label = self.annotations.iloc[idx]['annotation']
        sequence_path = os.path.join(self.frames_dir, video_id)
    
        # Check if the sequence folder exists
        if not os.path.exists(sequence_path):
            return None, None
    
        frames = []
        frame_files = sorted(os.listdir(sequence_path))
    
        # Check if there are no frames in the directory
        if not frame_files:
            return None, None
    
        # Apply temporal rescaling (only in training mode)
        if self.mode == 'train':
            frame_files = self.temporal_rescale(frame_files)
    
        # Load frames with error handling
        for frame_file in frame_files:
            frame_path = os.path.join(sequence_path, frame_file)
            try:
                frame = Image.open(frame_path)
                frame = self.transform(frame)
                frames.append(frame)
            except Exception as e:
                continue
    
        # If frames are empty after attempting to load, return None
        if not frames:
            return None, None
    
        frames_tensor = torch.stack(frames)
    
        # Convert label to tensor of indices
        label_tensor = torch.tensor([ord(c) - ord('a') for c in label], dtype=torch.long)
    
        return frames_tensor, label_tensor

    def temporal_rescale(self, frame_files):
        """
        Randomly rescale the number of frames in the sequence by a factor between 0.8 and 1.2.
        """
        rescale_factor = random.uniform(0.8, 1.2)
        original_length = len(frame_files)
        new_length = int(original_length * rescale_factor)
        indices = sorted(random.sample(range(original_length), new_length)) if new_length < original_length else \
                  [int(i * (original_length / new_length)) for i in range(new_length)]
        return [frame_files[i] for i in indices]

def create_dataloader(annotations_csv, frames_dir, batch_size=2, mode='train'):
    """
    Create a DataLoader for the PhoenixDataset with a custom collate function.
    """
    # Load annotations and print column names for debugging

    annotations = pd.read_csv(annotations_csv)


    # Ensure 'id' and 'annotation' columns are present
    if 'id' not in annotations.columns or 'annotation' not in annotations.columns:
        raise ValueError(f"The annotations CSV at {annotations_csv} must contain 'id' and 'annotation' columns.")

    # Create the dataset
    dataset = PhoenixDataset(annotations, frames_dir, mode)

    # Create the dataloader with the custom collate function
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(mode == 'train'), 
        num_workers=4, 
        collate_fn=custom_collate_fn
    )

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences by padding.
    Converts labels to tensors and pads inputs and labels.
    """
    # Filter out invalid samples
    batch = [sample for sample in batch if sample[0] is not None and sample[1] is not None]

    if len(batch) == 0:
        return None, None

    # Separate inputs and labels
    inputs, labels = zip(*batch)

    # Pad inputs to have the same number of frames
    inputs_padded = pad_sequence(inputs, batch_first=True)

    # Convert labels to tensors and pad them
    labels_tensors = [label.clone().detach().long() for label in labels]
    labels_padded = pad_sequence(labels_tensors, batch_first=True)

    return inputs_padded, labels_padded
