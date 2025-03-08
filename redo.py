import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import networkx as nx
import numpy as np

# --- Step 1: Generate Watts-Strogatz Graph ---
def generate_ws_graph(N, k, p):
    ws_graph = nx.watts_strogatz_graph(N, k, p)
    return ws_graph

# --- Step 2: Convert WS Graph to DAG (Directed Acyclic Graph) ---
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

# --- Step 3: Define MLP and LSTM Networks ---
class MLP(nn.Module):
    def __init__(self, structure):  # Accepts structure as an argument
        super(MLP, self).__init__()

        layers = []
        for i in range(len(structure) - 1):  # Loop through structure
            layers.append(nn.Linear(structure[i], structure[i + 1]))  # Linear layer
            if i < len(structure) - 2:  # No activation after last layer
                layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        return self.model(x)

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out

# --- Step 4: Implement Magnitude-Based Pruning ---
def magnitude_prune(model, prune_ratio):
    for name, param in model.named_parameters():
        if 'weight' in name:
            threshold = np.percentile(param.abs().detach().cpu().numpy(), prune_ratio * 100)
            mask = param.abs() > threshold
            param.data *= torch.tensor(mask, dtype=torch.float32, device=param.device)

# --- Step 5: Train the Models ---
import time

def train_model(model, train_loader, epochs=30, lr=0.001, log_interval=100):
    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optim.Adam(model.parameters(), lr=lr) # Optimizer Adam

    model.train()  # Set model to training mode

    for epoch in range(epochs):
        start_time = time.time()  # Track epoch time
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device) # Move to GPU

            optimizer.zero_grad()  # Reset gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_loss += loss.item()

            # Log every 'log_interval' batches
            if batch_idx % log_interval == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} completed - Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s\n")



# --- Load MNIST Dataset ---
device = torch.device("cuda")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # Normalize data. the numbers are the mean and std of the dataset (precalculated). The mean and std are used to normalize the data. the precalculated values are from the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# --- Apply Methods ---
N_mlp, k_mlp, p_mlp = 250, 2, [0.7, 0.8]
N_lstm, k_lstm, p_lstm = 48, [2, 4], [0.7, 0.8, 0.9, 1.0]
mlp_structure = [784, 89, 44, 22, 11, 4, 80]  # Input layer + hidden layers + output

lstm_structure = [16, 9, 6, 4, 2, 12]

# Generate WS Graph and DAG
ws_graph = generate_ws_graph(N_mlp, k_mlp, np.random.choice(p_mlp))
dag = undirected_to_dag(ws_graph)
#the dag structure can be used to make a neural network with the same structure as the dag 
# this is how:
# dag_structure = [len(dag.nodes)]
# for node in dag.nodes:
#     dag_structure.append(len(list(dag.successors(node))))
# print(dag_structure)
# this will give you the structure of the dag




# Initialize and Train MLP
mlp_model = MLP(mlp_structure).to(device)
train_model(mlp_model, train_loader)

# Prune and Retrain
for prune_ratio in np.arange(0, 1.05, 0.05):
    magnitude_prune(mlp_model, prune_ratio)
    train_model(mlp_model, train_loader)

print("Training and pruning completed!")
