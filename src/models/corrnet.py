import torch
import torch.nn as nn
from .resnet_backbone import ResNetBackbone
from .correlation_module import CorrelationModule
from .identification_module import CombinedIdentificationModule
from .temporal_model import ShortTermTemporalModule, LongTermTemporalModule

class CorrNetModel(nn.Module):
    def __init__(self, input_channels=3, hidden_size=1024, output_size=1000):
        super(CorrNetModel, self).__init__()

        # ResNet backbone
        self.backbone = ResNetBackbone(input_channels)

        self.correlation = CorrelationModule()

        # Linear layer to reduce channels to 3 for Short-term Temporal Module
        self.channel_reducer = nn.Linear(512, 3)

        # Short-term temporal model
        self.short_term_temporal = ShortTermTemporalModule(input_channels=3)

        # Long-term temporal model
        self.long_term_temporal = LongTermTemporalModule(
            input_size=512, hidden_size=hidden_size
        )

        # Identification module
        self.identification = CombinedIdentificationModule(input_channels)
        self.channel_downscale = nn.Conv3d(2048, 3, kernel_size=1)

        # Fully connected layer
        self.adjust_conv = nn.Conv3d(in_channels=3, out_channels=2048, kernel_size=1)
        self.fc = nn.Linear(2048, output_size)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size * num_frames, channels, height, width)

        # ResNet backbone
        processed_frames = self.backbone(x)
        processed_frames = processed_frames.view(batch_size, num_frames, -1)

        # Apply channel reduction to reduce processed_frames to 3 channels
        reduced_channels = self.channel_reducer(processed_frames)
        short_term_out = self.short_term_temporal(reduced_channels.permute(0, 2, 1))
        short_term_out = short_term_out.permute(0, 2, 1)

        # Long-term temporal modeling without channel reduction (keep at 512)
        long_term_out = self.long_term_temporal(processed_frames)
        prev_frame = short_term_out[:, :-2, :].unsqueeze(-1).unsqueeze(-1)
        current_frame = short_term_out[:, 1:-1, :].unsqueeze(-1).unsqueeze(-1)
        next_frame = short_term_out[:, 2:, :].unsqueeze(-1).unsqueeze(-1)

        correlated_features = self.correlation(prev_frame, current_frame, next_frame)
        long_term_out = long_term_out.unsqueeze(-1).unsqueeze(-1)
        downscaled_long_term_out = self.channel_downscale(long_term_out.permute(0, 2, 1, 3, 4))
        
        identified_features = self.identification(downscaled_long_term_out, correlated_features)
        adjusted_features = self.adjust_conv(identified_features)
        pooled_features = nn.AdaptiveAvgPool3d((1, 1, 1))(adjusted_features)
        flattened = pooled_features.view(pooled_features.size(0), -1)
        outputs = self.fc(flattened)

        return outputs
