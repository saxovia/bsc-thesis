import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Generate Watts-Strogatz Graph
def undirected_to_dag(G):
    adj_matrix = nx.to_numpy_array(G)
    n = adj_matrix.shape[0]
    dag = nx.DiGraph()
    dag.add_nodes_from(G.nodes)
    for i in range(n):
        for j in range(i):
            if adj_matrix[i][j] > 0:
                dag.add_edge(i, j)
    return dag

    
def generate_ws_graph(nodes, k=2, p=0.05):  # Adjusted p to match small-world properties
    return nx.watts_strogatz_graph(nodes, k, p)

def ws_to_dag(G):
    dag = nx.DiGraph()
    sorted_nodes = list(nx.topological_sort(G.to_directed()))
    layer_map = {}
    for node in sorted_nodes:
        predecessors = list(G.predecessors(node))
        if not predecessors:
            layer_map[node] = 0
        else:
            layer_map[node] = max(layer_map[p] for p in predecessors) + 1
        dag.add_node(node, layer=layer_map[node])
        for neighbor in G.neighbors(node):
            if layer_map[neighbor] > layer_map[node]:
                dag.add_edge(node, neighbor)
    return dag

def match_ws_to_mlp(G, layers=6):
    num_nodes = len(G.nodes)
    nodes_per_layer = np.array_split(sorted(G.nodes), layers)
    layer_sizes = [len(layer) for layer in nodes_per_layer]
    return [784] + layer_sizes + [10]

# Define MLP Model Class
class MLPNet(nn.Module):
    def __init__(self, structure):
        super().__init__()
        layers = []
        for i in range(len(structure) - 1):
            layers.append(nn.Linear(structure[i], structure[i + 1]))
            if i < len(structure) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)




# Define LSTM Model from WS Prior
def ws_to_lstm_structure(G):
    layers = max(nx.get_node_attributes(G, 'layer').values()) + 1
    hidden_sizes = [sum(1 for _, data in G.nodes(data=True) if data['layer'] == l) for l in range(layers)]
    return hidden_sizes

# Define LSTM Model Class
class LSTMNet(nn.Module):
    def __init__(self, hidden_sizes, output_dim=10):
        super().__init__()
        self.lstms = nn.ModuleList([nn.LSTM(hidden_sizes[i - 1] if i > 0 else 28, hidden_sizes[i], batch_first=True) for i in range(len(hidden_sizes))])
        self.fc = nn.Linear(hidden_sizes[-1], output_dim)
    
    def forward(self, x):
        for lstm in self.lstms:
            x, _ = lstm(x)
        return self.fc(x[:, -1, :])

# Magnitude-Based Pruning
def magnitude_prune(model, prune_ratio, mode="FULL"):
    for name, param in model.named_parameters():
        if 'weight' in name:
            threshold = np.percentile(param.abs().cpu().detach().numpy(), prune_ratio * 100)
            mask = param.abs() > threshold
            param.data *= mask.to(param.device).float()

# Training Function
def train(model, train_loader, epochs=30, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if isinstance(model, LSTMNet):
                images = images.view(-1, 28, 28)
            else:
                images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

# Load Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create and Train MLP with WS Prior
ws_graph = generate_ws_graph(250, k=2, p=0.05)
dag_graph = ws_to_dag(ws_graph)
mlp_structure = match_ws_to_mlp(dag_graph)
mlp_prior_model = MLPNet(mlp_structure)
train(mlp_prior_model, train_loader, epochs=30, lr=0.001)

# Train Fully Connected MLP Before Pruning
mlp_full_model = MLPNet([784, 256, 128, 64, 32, 10])
train(mlp_full_model, train_loader, epochs=30, lr=0.001)

# Prune and Train Again
prune_steps = np.arange(0, 1.05, 0.05)
for prune_ratio in prune_steps:
    magnitude_prune(mlp_full_model, prune_ratio, mode="FULL")
    train(mlp_full_model, train_loader, epochs=5, lr=0.001)

# Create and Train LSTM from WS Prior
lstm_structure = [sum(1 for _, data in dag_graph.nodes(data=True) if data['layer'] == l) for l in range(max(nx.get_node_attributes(dag_graph, 'layer').values()) + 1)]
lstm_structure = ws_to_lstm_structure(dag_graph)
lstm_prior_model = LSTMNet(hidden_sizes=lstm_structure)
train(lstm_prior_model, train_loader, epochs=50, lr=0.01)

# Train Fully Connected LSTM Before Pruning
lstm_full_model = LSTMNet(hidden_sizes=[128, 64, 32])
train(lstm_full_model, train_loader, epochs=50, lr=0.01)

# Apply LSTM-Specific Pruning in Steps
prune_steps_lstm = np.arange(0, 1.1, 0.1)
for prune_ratio in prune_steps_lstm:
    magnitude_prune(lstm_full_model, prune_ratio, mode="IH")
    train(lstm_full_model, train_loader, epochs=5, lr=0.01)
    magnitude_prune(lstm_full_model, prune_ratio, mode="HH")
    train(lstm_full_model, train_loader, epochs=5, lr=0.01)
