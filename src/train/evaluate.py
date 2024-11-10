import torch
from torch.utils.data import DataLoader
from src.models.corrnet import CorrNetModel
from src.utils.dataset import PhoenixDataset
from src.utils.metrics import calculate_wer

# Load dataset and model
val_data = ... # Load DataFrame with paths to validation videos
val_dataset = PhoenixDataset(val_data)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load trained model
corrnet = CorrNetModel(input_channels=3, hidden_size=1024, output_size=len(gloss_vocabulary))
checkpoint = torch.load("checkpoint.pth")
corrnet.load_state_dict(checkpoint['model_state_dict'])

# Evaluate the model
corrnet.eval()
predictions = []
references = []
for inputs in val_loader:
    with torch.no_grad():
        outputs = corrnet(inputs)
    # Perform decoding (e.g., greedy or beam search) to get predicted glosses
    # Append predictions and references for WER calculation

calculate_wer(predictions, references)
