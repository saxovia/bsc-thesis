import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Generate Watts-Strogatz Graph
def generate_ws_graph(nodes, k=4, p=0.1):
    return nx.watts_strogatz_graph(nodes, k, p)

# Convert WS Graph to MLP Structure
def ws_to_structure(G):
    layers = [784]  # MNIST Input layer
    num_nodes = len(G.nodes)
    avg_neighbors = max(5, int(np.mean([len(list(G.neighbors(n))) for n in G.nodes])))
    num_hidden_layers = max(2, min(5, num_nodes // avg_neighbors))
    hidden_layer_sizes = [max(16, avg_neighbors) for _ in range(num_hidden_layers)]
    layers.extend(hidden_layer_sizes)
    layers.append(10)  # Output layer
    return layers

# Match Fully Connected Network to WS Graph 
def match_ws_to_full(ws_graph):
    num_layers = len(set(nx.shortest_path_length(ws_graph, source=0).values()))
    nodes_per_layer = np.array_split(sorted(ws_graph.nodes), num_layers)
    layer_sizes = [len(layer) for layer in nodes_per_layer]
    layer_sizes[0] = 784  # Input Layer fix
    layer_sizes.append(10)  # Output Layer fix
    return layer_sizes

# Define MLP Model
def create_mlp(structure):
    layers = []
    for i in range(len(structure) - 1):
        layers.append(nn.Linear(structure[i], structure[i+1]))
        layers.append(nn.ReLU())
    layers.pop()  # Remove last ReLU
    return nn.Sequential(*layers)

# Define LSTM Model
class LSTMNet(nn.Module): #hid dim = 128 works
    def __init__(self, input_dim=28, hidden_dim=16, output_dim=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        #_, (h_n, _) = self.lstm(x)
        #return self.fc(h_n[-1])
        h0 = torch.zeros(1, x.size(0), 16).to(x.device)
        c0 = torch.zeros(1, x.size(0), 16).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# Magnitude-Based Pruning
def magnitude_prune(model, prune_ratio):
    for name, param in model.named_parameters():
        if 'weight' in name:
            threshold = np.percentile(param.abs().cpu().detach().numpy(), prune_ratio * 100)
            mask = param.abs() > threshold
            param.data *= torch.tensor(mask, dtype=torch.float32, device=param.device)

# LSTM Pruning (IH and HH)
def lstm_prune(model, prune_ratio, mode="IH"):
    for name, param in model.named_parameters():
        if ("weight_ih" in name and mode == "IH") or ("weight_hh" in name and mode == "HH"):
            threshold = np.percentile(param.abs().cpu().detach().numpy(), prune_ratio * 100)
            mask = (param.abs() > threshold).to(param.device).float()
            param.data *= mask  # Apply pruning mask
        
        elif mode == "FULL":
            # Get all weights in one array for global pruning
            all_weights = torch.cat([p.abs().flatten() for n, p in model.named_parameters() if "weight" in n])
            threshold = np.percentile(all_weights.cpu().detach().numpy(), prune_ratio * 100)

            # Apply global pruning
            mask = (param.abs() > threshold).to(param.device).float()
            param.data *= mask


#????
# Random Pruning
def random_prune(model, prune_ratio):
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask = torch.rand_like(param) > prune_ratio
            param.data *= mask.to(param.device)

# Training Function
def train(model, train_loader, epochs=10, lr=0.001, log_interval=100):
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        start_time = time.time()  # Track epoch time
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            if isinstance(model, LSTMNet):  # Reshape for LSTM
                images = images.view(-1, 28, 28)  # Reshape to (batch, seq_length=28, input_size=28)
            else:
                images = images.view(images.shape[0], -1)  # Flatten for MLP
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Log every 'log_interval' batches
            if batch_idx % log_interval == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} completed - Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s\n")
def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if isinstance(model, LSTMNet):  # Reshape for LSTM
                images = images.view(-1, 28, 28)  # Reshape to (batch, seq_length=28, input_size=28)
            else:
                images = images.view(images.shape[0], -1)  # Flatten for MLP
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# Load Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Data Loaded\n")

# Create and Train MLP with WS Prior
ws_graph = generate_ws_graph(200)
mlp_structure = ws_to_structure(ws_graph)
print(f"WS Graph: {mlp_structure}")
mlp_model = create_mlp(mlp_structure)
print("MLP Prior Model Created\n")
train(mlp_model, train_loader, epochs=10, lr=0.001)

print("MLP Prior Training Complete\n")
test_result = test(mlp_model, test_loader)   
test_performance = test(mlp_model, test_loader)
print(f"MLP Prior Test Performance: {test_performance}\n")


print("Pruning Model Training\n")
#

# Apply Pruning in Steps
mlp_structure = [784, 89, 44, 22, 11, 4, 80] 
fully_connected_nn = create_mlp(mlp_structure)
prune_steps_mlp = np.arange(0, 1.05, 0.05)  # 5% Steps for MLP
prune_steps_lstm = np.arange(0, 1.1, 0.1)  # 10% Steps for LSTM
#THIS PRUNING IS INCOMPLETE ACCORDING TO THE PAPER
for prune_ratio in prune_steps_mlp:
    magnitude_prune(fully_connected_nn, prune_ratio)
    train(fully_connected_nn, train_loader, epochs=5, lr=0.001)
    test_performance = test(fully_connected_nn, test_loader)
    test_result = test(fully_connected_nn, test_loader)
    print(f"Prune Ratio: {prune_ratio}, Test Performance: {test_performance}")
print("MLP Pruning Complete\n")

# Create and Train LSTM

lstm_model = LSTMNet()
print("LSTM Model Created\n")
train(lstm_model, train_loader, epochs=10, lr=0.01) # doesn't work
print("LSTM Training Complete\n")
# Apply LSTM-Specific Pruning in Steps
for prune_ratio in prune_steps_lstm:
    lstm_prune(lstm_model, prune_ratio, mode="IH")  # Prune IH first
    train(lstm_model, train_loader, epochs=5, lr=0.01)
    lstm_prune(lstm_model, prune_ratio, mode="HH")  # Then prune HH
    train(lstm_model, train_loader, epochs=5, lr=0.01)

print("LSTM Pruning Complete\n")