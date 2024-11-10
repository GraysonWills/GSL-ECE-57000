import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureReductionModule(nn.Module):
    def __init__(self, input_channels, reduction_factor=16):
        super(FeatureReductionModule, self).__init__()
        # Ensure reduced_channels is at least 1
        self.reduced_channels = max(1, input_channels // reduction_factor)
        self.conv = nn.Conv3d(input_channels, self.reduced_channels, kernel_size=1, bias=False)


    def forward(self, x):
        # Check if x is a list and concatenate along the depth (temporal) dimension
        if isinstance(x, list):
            x = torch.cat(x, dim=2)  # Concatenate along the depth dimension (dim=2)

        return self.conv(x)
class MultiScaleDilatedConvModule(nn.Module):
    def __init__(self, input_channels, n_scales=3, n_temporal_scales=4):
        super(MultiScaleDilatedConvModule, self).__init__()
        
        # Ensure the groups parameter is valid
        groups = max(1, input_channels)

        # Create parallel convolutional branches with increasing dilation rates
        self.branches = nn.ModuleList([
            nn.Conv3d(
                input_channels, input_channels, 
                kernel_size=(3, 3, 3), 
                dilation=(1, 2**i, 2**j), 
                padding=(1, 2**i, 2**j), 
                groups=groups  # Use valid groups for depthwise convolution
            )
            for i in range(n_scales) for j in range(n_temporal_scales)
        ])
        
        # Learnable coefficients for each branch
        self.coefficients = nn.Parameter(torch.ones(n_scales * n_temporal_scales))

    def forward(self, x):
        outputs = []
        for i, branch in enumerate(self.branches):
            outputs.append(self.coefficients[i] * branch(x))
        
        # Sum the outputs of all branches
        return sum(outputs)

import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentificationModule(nn.Module):
    def __init__(self, input_channels, reduction_factor=16):
        super(IdentificationModule, self).__init__()
        
        self.channel_adjust = nn.Conv3d(input_channels, 3, kernel_size=1, stride=1)
        self.reduction = FeatureReductionModule(3, reduction_factor)
        reduced_channels = self.reduction.reduced_channels
        self.multi_scale = MultiScaleDilatedConvModule(reduced_channels)
        self.output_conv = nn.Conv3d(reduced_channels, input_channels, kernel_size=1, bias=False)

    def forward(self, x, T):
        # Ensure T has an extra depth dimension if it doesn’t already
        if T.dim() == 4:  
            T = T.unsqueeze(2)  


        # Resize T’s temporal dimension to match x’s
        T = F.interpolate(T, size=(x.size(2), x.size(3), x.size(4)), mode='trilinear', align_corners=True)
     

        # Match T’s batch size to x’s batch size if different
        if T.size(0) != x.size(0):
            if T.size(0) == 1:  # Case where T is a single batch; expand to match x
                T = T.expand(x.size(0), -1, -1, -1, -1)
            else:
                # If batch sizes don’t match in a meaningful way, try reducing T's batch size
                T = T[:x.size(0)]


        # Adjust T’s channels if necessary
        if T.size(1) != x.size(1):
            T = self.channel_adjust(T)


        # Proceed with feature reduction and multi-scale convolutions
        reduced_x = self.reduction(x)

        
        multi_scale_output = self.multi_scale(reduced_x)


        attention_map = torch.sigmoid(self.output_conv(multi_scale_output)) - 0.5
        attention_map_resized = F.interpolate(attention_map, size=T.shape[2:], mode='trilinear', align_corners=True)
        
        # Apply attention map to T
        output = x + T * attention_map_resized


        return output

class CombinedIdentificationModule(nn.Module):
    def __init__(self, input_channels, reduction_factor=16):
        super(CombinedIdentificationModule, self).__init__()
        # Learnable alpha parameter, initialized to 0.5
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Identification Module with feature reduction and multi-scale convolutions
        self.identification = IdentificationModule(input_channels, reduction_factor)

    def forward(self, x, T):
        """
        x: Original frame features
        T: Correlated features from Correlation Module
        """
        # Debug shapes of x and T


        # Apply the identification module to the correlated features
        M = self.identification(x, T)


        # Blend original and modulated features using alpha
        output = x + self.alpha * M

        return output
