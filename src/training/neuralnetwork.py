import torch.nn as nn
import torch
import torch.nn.functional as F

class MLPNet(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            nn.init.xavier_normal_(linear.weight)
            nn.init.constant_(linear.bias, 0)
            layers.append(linear)
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_dim=10):
        super().__init__()
        self.lstms = nn.ModuleList([nn.LSTM(
            hidden_sizes[i - 1] if i > 0 else input_size,
            hidden_sizes[i], 
            batch_first=True) 
            for i in range(len(hidden_sizes))])
        self.fc = nn.Linear(hidden_sizes[-1], output_dim)
        for lstm in self.lstms:
            for name, param in lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, x):
        for lstm in self.lstms:
            x, _ = lstm(x)
        return self.fc(x[:, -1, :])
    

import torch
import torch.nn as nn
import networkx as nx

class SparseMLPNet(nn.Module):
    def __init__(self, layer_sizes, hidden_sizes=None, dag_graph=None):
        super().__init__()
        if not isinstance(layer_sizes, list):
            layer_sizes = [layer_sizes[0]] + (hidden_sizes if hidden_sizes else []) + [layer_sizes[-1]]
        
        self.layer_sizes = layer_sizes
        self.dag_graph = dag_graph  # DAG representing the network structure
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            nn.init.xavier_normal_(linear.weight)
            nn.init.constant_(linear.bias, 0)
            layers.append(linear)
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        activations = {f'input_{i}': x[:, i] for i in range(self.layer_sizes[0])}

        for layer_idx in range(1, len(self.layer_sizes)):
            next_activations = {}
            for neuron_idx in range(self.layer_sizes[layer_idx]):
                total_input = 0
                for prev_neuron_idx in range(self.layer_sizes[layer_idx - 1]):
                    if self.dag_graph.has_edge(f'input_{prev_neuron_idx}', f'layer{layer_idx}_{neuron_idx}'):
                        total_input += activations[f'input_{prev_neuron_idx}'] * self.network[layer_idx - 1].weight[prev_neuron_idx, neuron_idx]
                next_activations[f'layer{layer_idx}_{neuron_idx}'] = torch.relu(total_input)
            activations.update(next_activations)
        return activations[f'layer{len(self.layer_sizes) - 1}_0']


class SparseLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_dim=10, dag_graph=None):
        super().__init__()
        
        if not isinstance(hidden_sizes, (list, tuple)):
            hidden_sizes = [hidden_sizes]
        
        self.dag_graph = dag_graph
        self.lstms = nn.ModuleList()
        current_input_size = input_size
        for hidden_size in hidden_sizes:
            self.lstms.append(nn.LSTM(
                input_size=current_input_size,
                hidden_size=hidden_size,
                batch_first=True
            ))
            current_input_size = hidden_size
        self.fc = nn.Linear(current_input_size, output_dim)

    def forward(self, x):
        sparse_input = []
        for i in range(x.size(1)):
            if self.dag_graph.has_edge(f'input_{i}', 'input_sparsified'):
                sparse_input.append(x[:, i])
        
        sparse_input = torch.stack(sparse_input, dim=1)
        
        for lstm in self.lstms:
            sparse_input, _ = lstm(sparse_input)
        
        return self.fc(sparse_input[:, -1, :])