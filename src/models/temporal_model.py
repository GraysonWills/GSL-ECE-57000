import torch
import torch.nn as nn

class ShortTermTemporalModule(nn.Module):
    def __init__(self, input_channels):
        super(ShortTermTemporalModule, self).__init__()
        self.conv = nn.Conv1d(input_channels, input_channels, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        return self.conv(x)


class LongTermTemporalModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LongTermTemporalModule, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):
        output, _ = self.lstm(x)
        return output
