import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self, input_channels=3):
        super(ResNetBackbone, self).__init__()
        resnet18 = models.resnet18(pretrained=True)

        # Modify the first layer to match input_channels
        if input_channels != 3:
            resnet18.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.feature_extractor = nn.Sequential(*list(resnet18.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.squeeze(-1).squeeze(-1)  # Remove extra dimensions
