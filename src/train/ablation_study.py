import torch
from torch.utils.data import DataLoader
from src.models.corrnet import CorrNetModel
from src.models.correlation_module import CorrelationModule
from src.models.identification_module import CombinedIdentificationModule
from src.utils.dataset import PhoenixDataset
from src.utils.training_utils import train_model, create_optimizer_and_scheduler

def create_ablation_model(enable_correlation=True, enable_identification=True):
    modules = []
    modules.append(CorrelationModule() if enable_correlation else torch.nn.Identity())
    modules.append(CombinedIdentificationModule(input_channels=3) if enable_identification else torch.nn.Identity())
    return torch.nn.Sequential(*modules)

# Load dataset
train_data = ...  # Load the DataFrame with paths to training videos
train_dataset = PhoenixDataset(train_data, transform=None)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Conduct ablation study
ablations = [
    {"enable_correlation": True, "enable_identification": True},
    {"enable_correlation": False, "enable_identification": True},
    {"enable_correlation": True, "enable_identification": False},
]

for ablation in ablations:
    model = create_ablation_model(**ablation)
    optimizer, scheduler = create_optimizer_and_scheduler(model)
    criterion = torch.nn.CTCLoss()
    
    print(f"Running ablation: {ablation}")
    train_model(model, train_loader, optimizer, criterion, num_epochs=10)
