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
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)


    def __init__(self, model_type, dataset_type, hidden_sizes=None, lr=0.001, loss="CrossEntropy", optimizer="Adam", epochs=30, k=2, p=0.05):
        super().__init__()
        if model_type == "MLP":
            self.model = MLPNet(hidden_sizes)
        elif model_type == "LSTM":
            self.model = LSTMNet(hidden_sizes)
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.k = k
        self.p = p
        self.dataset_type = dataset_type
        #self.train_loader, self.test_loader = self.load_data()


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #loss function
        if loss == "CrossEntropy":
            self.criterion = nn.CrossEntropyLoss()
        else: #for now default to MSELoss
            self.criterion = nn.MSELoss()


        self.running = True

    def run(self):
        if self.dataset_type == "MNIST":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
            test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
        elif self.dataset_type == "CIFAR-10":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
            test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
        elif self.dataset_type == "CIFAR-100":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_dataset = datasets.CIFAR100(root="./data", train=True, transform=transform, download=True)
            test_dataset = datasets.CIFAR100(root="./data", train=False, transform=transform, download=True)
        else:
            raise ValueError("Invalid dataset type")
        print(f"Loaded {self.dataset_type} dataset.")
        progress_message = f"Loaded {self.dataset_type} dataset."
        self.message.emit(progress_message)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        ws_graph = self.generate_ws_graph(250, k=self.k, p=self.p)
        dag_graph = self.ws_to_dag(ws_graph)
        if self.model.__class__.__name__ == "LSTMNet":
            lstm_structure = self.ws_to_lstm_structure(dag_graph)
            self.model = LSTMNet(lstm_structure).to(self.device)
            print("LSTM NN created.")
            progress_message = "LSTM NN created."
            self.message.emit(progress_message)
        else:
            mlp_structure = self.match_ws_to_mlp(dag_graph)
            self.model = MLPNet(mlp_structure).to(self.device)
            print("MLP NN created.")
            progress_message = "MLP NN created."
            self.message.emit(progress_message)

        print(f"Parameters of the trainer: {self.epochs} epoch, {self.lr} learning_rate, {self.optimizer} optimizer, {self.criterion} loss function, {self.k} k, {self.p} p, {self.dataset_type} dataset.")
        progress_message = f"Parameters of the trainer: {self.epochs} epoch, {self.lr} learning_rate, {self.optimizer} optimizer, {self.criterion} loss function, {self.k} k, {self.p} p, {self.dataset_type} dataset."
        self.message.emit(progress_message)


        print("Starting the training process...")
        progress_message = "Starting the training process..."
        self.message.emit(progress_message)

        self.train(self.model, train_loader, self.epochs, lr=self.lr)
        self.finished.emit()  # notify gui when done

    
    def stop(self):
        """Method to stop training from the GUI"""
        self.running = False
        self.finished.emit()


    def generate_ws_graph(self, nodes, k=2, p=0.05):
        return nx.watts_strogatz_graph(nodes, k, p)

    def train(self, model, train_loader, epochs=30, lr=0.001):
        model.to(self.device)
        criterion = self.criterion
        if self.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif self.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif self.optimizer == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        elif self.optimizer == "Adadelta":
            optimizer = optim.Adadelta(model.parameters(), lr=lr)
        elif self.optimizer == "Adagrad":
            optimizer = optim.Adagrad(model.parameters(), lr=lr)
        else:  
            optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()

        epoch = 0  # Track epoch manually
        while self.running and epoch < epochs:  # Stop instantly when self.running is False
            total_loss = 0
            correct = 0
            total = 0

            for images, labels in train_loader:
                if not self.running:  # **Exit immediately if stop() is called**
                    print("Stopping training early...")
                    return  

                images, labels = images.to(self.device), labels.to(self.device)
                if isinstance(model, LSTMNet):
                    images = images.view(-1, 28, 28)
                else:
                    images = images.view(images.shape[0], -1)

                optimizer.zero_grad()
                outputs = model(images)

                if isinstance(criterion, nn.MSELoss):
                    labels_one_hot = torch.zeros(labels.size(0), 10).to(self.device)
                    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
                    loss = criterion(outputs, labels_one_hot)
                else:
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%")
            self.message.emit(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%")
            
            epoch += 1  # Increment epoch counter manually

        print("Training complete.")
        self.message.emit("Training complete.")


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