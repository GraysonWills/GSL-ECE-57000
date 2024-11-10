import torch
from src.models.corrnet import CorrNetModel
from src.utils.dataset import create_dataloader
from src.utils.training_utils import (
    create_optimizer_and_scheduler, 
    save_checkpoint, 
    load_checkpoint, 
    train_one_epoch, 
    validate_model
)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, start_epoch=0, num_epochs=20, device='cuda'):
    model.to(device)

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")

        # Validation
        val_loss = validate_model(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Step the learning rate scheduler
        scheduler.step()

        # Save checkpoint at the end of each epoch
        save_checkpoint(model, optimizer, epoch + 1)

if __name__ == '__main__':
    # Define file paths
    train_csv = "data/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/train.corpus.csv"
    val_csv = "data/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/dev.corpus.csv"
    train_frames_dir = 'data/output_preprocess'
    dev_frames_dir = 'data/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/dev'

    # Load data
    train_loader = create_dataloader(train_csv, train_frames_dir, batch_size=2, mode='train')
    val_loader = create_dataloader(val_csv, dev_frames_dir, batch_size=2, mode='val')

    # Initialize model, criterion, optimizer, and scheduler
    model = CorrNetModel(input_channels=3, hidden_size=1024, output_size=1000)  # Adjust output size based on vocabulary size
    criterion = torch.nn.CTCLoss(blank=0)  # Adjust blank token as necessary
    optimizer, scheduler = create_optimizer_and_scheduler(model)

    # Load from a checkpoint if available
    start_epoch = load_checkpoint(model, optimizer)

    # Start training
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, start_epoch)
