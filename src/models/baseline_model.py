import torch

class BaselineModel(torch.nn.Module):
    def __init__(self, input_channels, hidden_size, output_size):
        super(BaselineModel, self).__init__()
        self.cnn = torch.nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.lstm = torch.nn.LSTM(64, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        cnn_features = self.cnn(x)
        lstm_output, _ = self.lstm(cnn_features)
        output = self.fc(lstm_output)
        return output
