import torch
class CorrelationModule(torch.nn.Module):
    def __init__(self):
        super(CorrelationModule, self).__init__()
        self.beta1 = torch.nn.Parameter(torch.tensor(0.5))
        self.beta2 = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, prev_frame, current_frame, next_frame):
        # Determine dimensions based on next_frame's structure
        if next_frame.dim() == 5:  # Case: Batch, Temporal, Channel, Height, Width
            batch_size, temporal_dim, num_channels, height, width = next_frame.size()
            # Use reshape instead of view
            next_frame = next_frame.reshape(batch_size * temporal_dim, num_channels, height, width)
            current_frame = current_frame.reshape(batch_size * temporal_dim, num_channels, height, width)
            prev_frame = prev_frame.reshape(batch_size * temporal_dim, num_channels, height, width)
            batch_size = batch_size * temporal_dim
        elif next_frame.dim() == 4:  # Case: Batch, Channel, Height, Width
            batch_size, num_channels, height, width = next_frame.size()
        elif next_frame.dim() == 3:  # Case: Batch, Channel, Height (Width is 1)
            batch_size, num_channels, height = next_frame.size()
            width = 1
        else:
            raise ValueError(f"Unexpected next_frame dimensions: {next_frame.dim()}")


        # Compute forward correlation (current to next)
        forward_corr = self.compute_correlation_map(current_frame, next_frame, batch_size, num_channels)
        refined_forward = self.refine_correlation(forward_corr, batch_size, num_channels, height, width)

        # Compute backward correlation (current to prev)
        backward_corr = self.compute_correlation_map(current_frame, prev_frame, batch_size, num_channels)
        refined_backward = self.refine_correlation(backward_corr, batch_size, num_channels, height, width)

        # Combine forward and backward correlations
        T = self.beta1 * refined_forward + self.beta2 * refined_backward
        return T

    def compute_correlation_map(self, frame1, frame2, batch_size, num_channels):
        correlation_map = torch.einsum('bchw,bchw->bhw', frame1, frame2) / num_channels
        correlation_map = correlation_map.unsqueeze(1).expand(batch_size, num_channels, *correlation_map.shape[-2:])
        return correlation_map

    def refine_correlation(self, correlation_map, batch_size, num_channels, height, width):
        refined_map = torch.sigmoid(correlation_map) - 0.5
        if refined_map.dim() == 3:
            refined_map = refined_map.unsqueeze(1)
        
        # Expand refined_map to match the num_channels dimension
        refined_map = refined_map.expand(batch_size, num_channels, height, width)
        return refined_map
