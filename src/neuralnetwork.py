import torch.nn as nn

class MLPNet(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class LSTMNet(nn.Module):
    def __init__(self, hidden_sizes, output_dim=10):
        super().__init__()
        self.lstms = nn.ModuleList([nn.LSTM(hidden_sizes[i - 1] if i > 0 else 28, hidden_sizes[i], batch_first=True) for i in range(len(hidden_sizes))])
        self.fc = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x):
        for lstm in self.lstms:
            x, _ = lstm(x)
        return self.fc(x[:, -1, :])