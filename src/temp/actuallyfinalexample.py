import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Generate Watts-Strogatz Graph
def generate_ws_graph(nodes, k=2, p=0.05):
    return nx.watts_strogatz_graph(nodes, k, p)

# Convert WS Graph to DAG
def ws_to_dag(G):
    adj_matrix = nx.to_numpy_array(G)
    n = adj_matrix.shape[0]
    dag = nx.DiGraph()
    dag.add_nodes_from(G.nodes)
    for i in range(n):
        for j in range(i):
            if adj_matrix[i][j] > 0:
                dag.add_edge(i, j)
    layers = {}
    for node in nx.topological_sort(dag):
        predecessors = list(dag.predecessors(node))
        layers[node] = max((layers[p] for p in predecessors), default=-1) + 1
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
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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

# Load Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print("MLP PRIOR")
ws_graph = generate_ws_graph(250, k=2, p=0.05)
dag_graph = ws_to_dag(ws_graph)
mlp_structure = match_ws_to_mlp(dag_graph)
mlp_prior_model = create_mlp(mlp_structure)
train(mlp_prior_model, train_loader, epochs=30, lr=0.001)

print("MLP Pruning 1st")
mlp_full_model = create_mlp([784, 256, 128, 64, 32, 10])
train(mlp_full_model, train_loader, epochs=3, lr=0.001)

print("MLP Pruning 2nd")
prune_steps = np.arange(0, 1.05, 0.05)
for prune_ratio in prune_steps:
    magnitude_prune(mlp_full_model, prune_ratio, mode="FULL")
    train(mlp_full_model, train_loader, epochs=5, lr=0.001)
    print("Iteration complete for: ", prune_ratio)
