from src.neuralnetwork import MLPNet, LSTMNet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PyQt6.QtCore import QThread, pyqtSignal
# Define the Trainer class


class Trainer(QThread):
    progress = pyqtSignal(int)  # Signal to update progress bar
    finished = pyqtSignal()  # Signal when training is complete
    message = pyqtSignal(str)  # Status messages


    def __init__(self, model_type, dataset_type, hidden_sizes=None, lr=0.001):
        super().__init__()
        if model_type == "MLP":
            self.model = MLPNet(hidden_sizes)
        elif model_type == "LSTM":
            self.model = LSTMNet(hidden_sizes)
        self.lr = lr
        self.dataset_type = dataset_type
        #self.train_loader, self.test_loader = self.load_data()


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.running = True

    def run(self):  # This method runs in a separate thread when `.start()` is called
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        ws_graph = self.generate_ws_graph(250, k=2, p=0.05)
        dag_graph = self.ws_to_dag(ws_graph)
        mlp_structure = self.match_ws_to_mlp(dag_graph)
        self.model = MLPNet(mlp_structure).to(self.device)

        self.train(self.model, train_loader, epochs=30, lr=self.lr)
        self.finished.emit()  # Notify GUI when training is complete

    
    def stop(self):
        """Method to stop training from the GUI"""
        self.running = False

    def generate_ws_graph(self, nodes, k=2, p=0.05):
        return nx.watts_strogatz_graph(nodes, k, p)

    def train(self, model, train_loader, epochs=30, lr=0.001):
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()

        for epoch in range(epochs):
            if not self.running:
                break
            total_loss = 0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
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
            progress_message = f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%"
            self.message.emit(progress_message) 


    def match_ws_to_mlp(self, G, layers=6):
        num_nodes = len(G.nodes)
        nodes_per_layer = np.array_split(sorted(G.nodes), layers)
        layer_sizes = [len(layer) for layer in nodes_per_layer]
        return [784] + layer_sizes + [10]


    def ws_to_dag(self, G):
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