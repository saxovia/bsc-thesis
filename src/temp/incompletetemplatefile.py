
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def generate_ws_graph(nodes, k=2, p=0.05): 
    return nx.watts_strogatz_graph(nodes, k, p)


def ws_to_dag(G):
    adj_matrix = nx.to_numpy_array(G)
    n = adj_matrix.shape[0]
    dag = nx.DiGraph()
    dag.add_nodes_from(G.nodes)
    for i in range(n):
        for j in range(i): #ensures that the lower triangular matrix is used
            if adj_matrix[i][j] > 0: #checks if there is an edge between i and j
                dag.add_edge(i,j)
    layers = {}
    for node in nx.topological_sort(dag):
        predecessors = list(dag.predecessors(node))
        if predecessors:
            layers[node] = max(layers[p] for p in predecessors) + 1
        else:
            layers[node] = 0 #root

    nx.set_node_attributes(dag, layers, 'layer')

    return dag
# Match WS DAG to MLP Structure
def match_ws_to_mlp(G, layers=6):
    num_nodes = len(G.nodes)
    nodes_per_layer = np.array_split(sorted(G.nodes), layers)
    layer_sizes = [len(layer) for layer in nodes_per_layer]
    return [784] + layer_sizes + [10]

# Define MLP Model
def create_mlp(structure):
    layers = []
    for i in range(len(structure) - 1):
        layers.append(nn.Linear(structure[i], structure[i+1]))
        if i < len(structure) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def ws_to_lstm_structure(G):
    layers = max(nx.get_node_attributes(G, 'layer').values()) + 1
    hidden_sizes = [sum(1 for _, data in G.nodes(data=True) if data['layer'] == l) for l in range(layers)]
    return hidden_sizes
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


def magnitude_prune(model, prune_ratio, mode="FULL"):
    for name, param in model.named_parameters():
        if 'weight' in name:
            threshold = np.percentile(param.abs().cpu().detach().numpy(), prune_ratio * 100)
            mask = param.abs() > threshold
            param.data *= mask.to(param.device).float()

def train(model, train_loader, epochs=30, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if isinstance(model, LSTMNet):
                images = images.view(-1, 28, 28)
            else:
                images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%")


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("LSTM Pruning1st")
lstm_full_model = LSTMNet(hidden_sizes=[128, 64, 32])
train(lstm_full_model, train_loader, epochs=3, lr=0.01)

print("LSTM Pruning2nd")
prune_steps_lstm = np.arange(0, 1.1, 0.1)
for prune_ratio in prune_steps_lstm:
    magnitude_prune(lstm_full_model, prune_ratio, mode="IH")
    train(lstm_full_model, train_loader, epochs=5, lr=0.01)
    print("IH Iteration complete for: ", prune_ratio)
    magnitude_prune(lstm_full_model, prune_ratio, mode="HH")
    train(lstm_full_model, train_loader, epochs=5, lr=0.01)
    print("HH Iteration complete for: ", prune_ratio)


print("MLP PRIOR")
ws_graph = generate_ws_graph(250, k=2, p=0.05)
dag_graph = ws_to_dag(ws_graph)
mlp_structure = match_ws_to_mlp(dag_graph)
mlp_prior_model = MLPNet(mlp_structure)
train(mlp_prior_model, train_loader, epochs=30, lr=0.001)
print("LSTM PRIOR")
lstm_structure = ws_to_lstm_structure(dag_graph)
lstm_prior_model = LSTMNet(hidden_sizes=lstm_structure)
train(lstm_prior_model, train_loader, epochs=30, lr=0.01)



print("MLP Pruning 1st")
mlp_full_model = create_mlp([784, 256, 128, 64, 32, 10])
train(mlp_full_model, train_loader, epochs=3, lr=0.001)
print("MLP Pruning 2nd")

prune_steps = np.arange(0, 1.05, 0.05)
for prune_ratio in prune_steps:
    magnitude_prune(mlp_full_model, prune_ratio, mode="FULL")
    train(mlp_full_model, train_loader, epochs=5, lr=0.001)
    print("Iteration complete for: ", prune_ratio)







##### MAYBE

def get_prune_mask(weight_tensor, prune_percent):
    weight_np = weight_tensor.cpu().detach().numpy().flatten()
    threshold = np.percentile(np.abs(weight_np), prune_percent)
    mask = (torch.abs(weight_tensor) > threshold).float()
    return mask

def magnitude_prune(model, prune_percent, mode="FULL"):
    if isinstance(model, LSTMNet):
        # LSTM pruning logic
        for name, param in model.named_parameters():
            if "weight" in name:  # Ignore biases
                if mode == "FULL":
                    mask = get_prune_mask(param, prune_percent)
                    param.data.mul_(mask)
                elif mode == "IH" and "weight_ih" in name:
                    mask = get_prune_mask(param, prune_percent)
                    param.data.mul_(mask)
                elif mode == "HH" and "weight_hh" in name:
                    mask = get_prune_mask(param, prune_percent)
                    param.data.mul_(mask)
                elif mode == "HTO" and hasattr(model, "fc") and "fc.weight" in name:
                    mask = get_prune_mask(param, prune_percent)
                    param.data.mul_(mask)

    elif isinstance(model, MLPNet) or isinstance(model, nn.Sequential):
        # MLP pruning logic
        layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
        
        if mode == "FULL":
            for layer in layers:
                mask = get_prune_mask(layer.weight, prune_percent)
                layer.weight.data.mul_(mask)
        
        elif mode == "IH" and len(layers) > 0:
            # Prune only the first layer
            mask = get_prune_mask(layers[0].weight, prune_percent)
            layers[0].weight.data.mul_(mask)

        elif mode == "HH" and len(layers) > 2:
            # Prune hidden layers (all except first and last)
            for layer in layers[1:-1]:
                mask = get_prune_mask(layer.weight, prune_percent)
                layer.weight.data.mul_(mask)

        elif mode == "HTO" and len(layers) > 0:
            # Prune only the last layer
            mask = get_prune_mask(layers[-1].weight, prune_percent)
            layers[-1].weight.data.mul_(mask)
