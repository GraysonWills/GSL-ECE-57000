import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os

def create_optimizer_and_scheduler(model, learning_rate=0.001, step_size=10, gamma=0.5):
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    return optimizer, scheduler

def save_checkpoint(model, optimizer, epoch, path="experiments/logs/checkpoints/corrnet_checkpoint.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path="experiments/logs/checkpoints/corrnet_checkpoint.pth"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    # Wrap the DataLoader in tqdm to display a progress bar
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False)
    
    for batch_idx, (inputs, labels) in progress_bar:
        if inputs is None or labels is None:
            # Skip this batch if any element is None
            continue
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Compute `target_lengths` based on labels
        target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long).to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)  # Expected shape: [batch_size, num_classes]
        
        # Reshape `outputs` to [seq_length, batch_size, num_classes]
        outputs = outputs.unsqueeze(0)  # Add sequence length dimension if necessary
        
        # Set `input_lengths` based on outputs' sequence length
        batch_size = outputs.size(1)
        seq_length = outputs.size(0)
        input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long).to(device)
        
        # Compute CTC Loss
        loss = criterion(outputs, labels, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()
        
        # Update progress bar with the current batch loss
        progress_bar.set_postfix(loss=loss.item())
    
    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(train_loader)
    print(f"\n[Epoch Summary] Average Training Loss: {epoch_loss:.4f}")

    return epoch_loss


def validate_model(model, dataloader, criterion, device='cuda'):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Adjust input_lengths to reflect num_frames - 2
            input_lengths = torch.full((inputs.size(0),), inputs.size(1) - 2, dtype=torch.long)
            target_lengths = torch.full((labels.size(0),), labels.size(1), dtype=torch.long)

            # Forward pass
            outputs = model(inputs)

            # CTC loss calculation
            loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, target_lengths)

            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss
