from src.neuralnetwork import MLPNet, LSTMNet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PyQt6.QtCore import QThread, pyqtSignal
from src.pruner import MagnitudePruner, RandomPruner, L1Pruner
# Define the Trainer class


class Trainer(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)


    def __init__(self, model_type, dataset_type, lr=None, hidden_sizes=[6,6,6], loss="CrossEntropy", optimizer="Adam", epochs=30, k=2, p=0.05,graph_type="Full", N=250, batch_size=64):

        super().__init__()
        if hidden_sizes == '':
            self.hidden_sizes = [6, 6, 6]
        elif isinstance(hidden_sizes, str):
            self.hidden_sizes = [int(x) for x in hidden_sizes.split(",")]
        else:
            self.hidden_sizes = hidden_sizes
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
        self.graph_type = graph_type
        self.N = int(N)
        self.batch_size = batch_size
        #self.train_loader, self.test_loader = self.load_data()

        self.prune_self = {
            "Magnitude": MagnitudePruner(),
            "Random": RandomPruner(),
            "L1": L1Pruner()
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #loss function
        if loss == "CrossEntropy":
            self.criterion = nn.CrossEntropyLoss()
        else: #for now default to MSELoss
            self.criterion = nn.MSELoss()


        self.running = True
    def load_data_and_create_graph(self):
        dataset_info = {
            "MNIST": {
                "dataset": datasets.MNIST,
                "input_size": 28 * 28,
                "num_classes": 10,
                "transform": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                "default_batch_size": 64
            },
            "CIFAR-10": {
                "dataset": datasets.CIFAR10,
                "input_size": 3 * 32 * 32,
                "num_classes": 10,
                "transform": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                "default_batch_size": 32
            },
            "CIFAR-100": {
                "dataset": datasets.CIFAR100,
                "input_size": 3 * 32 * 32,
                "num_classes": 100,
                "transform": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                "default_batch_size": 128 #TODO Adjust every default value assignment because it is all over the place
            }
        }

        dataset_config = dataset_info[self.dataset_type]
        input_size = dataset_config["input_size"]
        num_classes = dataset_config["num_classes"]
        transform = dataset_config["transform"]
        default_batch_size = dataset_config["default_batch_size"]
        batch_size = self.batch_size if self.batch_size is not None else default_batch_size


        train_dataset = dataset_config["dataset"](root="./data", train=True, transform=transform, download=True)
        test_dataset = dataset_config["dataset"](root="./data", train=False, transform=transform, download=True)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


        print(f"Loaded {self.dataset_type} dataset.")
        progress_message = f"Loaded {self.dataset_type} dataset."
        self.message.emit(progress_message)

        dag_graph = None
        if self.graph_type == "WS":
            ws_graph = self.generate_ws_graph(self.N, k=self.k, p=self.p)
            dag_graph = self.ws_to_dag(ws_graph)
        elif self.graph_type == "Full":
            dag_graph = self.generate_fully_connected_graph(self.hidden_sizes[0]) #TODO Change later!!


        # Convert DAG to NN
        if self.model.__class__.__name__ == "LSTMNet":
            lstm_structure = self.dag_to_lstm_structure(dag_graph, input_size, num_classes)
            self.model = LSTMNet(lstm_structure).to(self.device)
            print("LSTM NN created.")
            self.message.emit("LSTM NN created.")
        else:
            mlp_structure = self.dag_to_mlp_structure(dag_graph, input_size, num_classes)
            self.model = MLPNet(mlp_structure).to(self.device)
            print("MLP NN created.")
            self.message.emit("MLP NN created.")
        if self.lr != None:
            print(f"Parameters of the trainer: {self.epochs} epoch, {self.lr} learning_rate, {self.optimizer} optimizer, {self.criterion} loss function, {self.k} k, {self.p} p, {self.dataset_type} dataset, {self.N} N.")
        else: 
            print(f"Parameters of the trainer: {self.epochs} epoch, default learning_rate, {self.optimizer} optimizer, {self.criterion} loss function, {self.k} k, {self.p} p, {self.dataset_type} dataset, {self.N} N.")
        progress_message = f"Parameters of the trainer: {self.epochs} epoch, {self.lr} learning_rate, {self.optimizer} optimizer, {self.criterion} loss function, {self.k} k, {self.p} p, {self.dataset_type} dataset, {self.N} N."
        self.message.emit(progress_message)



    def run(self):
        print("Starting the training process...")
        progress_message = "Starting the training process..."
        self.message.emit(progress_message)

        self.train(self.model, self.train_loader, self.epochs, lr=self.lr)
        self.finished.emit()  # notify gui when done


    
    def stop(self):
        self.running = False
        self.finished.emit()
    def generate_fully_connected_graph(self, nodes):
        """
        Generates a fully connected directed acyclic graph (DAG).

        Args:
            nodes (int): Number of nodes in the graph.

        Returns:
            nx.DiGraph: A fully connected DAG.
        """
        G = nx.complete_graph(nodes, create_using=nx.DiGraph)
        
        # Ensure the graph is acyclic by directing edges from smaller index to larger index
        for u, v in list(G.edges):
            if u > v:
                G.remove_edge(u, v)
        
        layers = {node: node for node in G.nodes()}  # Simple layer assignment
        nx.set_node_attributes(G, layers, 'layer')

        return G

    def prune_self(self, layer,  prune_ratio, prune_type):
        for prune_types in self.prune_handler:
            if prune_type == "Magnitude":
                self.magnitude_prune(self.model, prune_ratio=0.5, mode="FULL")
            elif prune_type == "Random":
                self.random_prune(self.model, prune_ratio=0.5)
            elif prune_type == "Structured":
                self.structured_prune(self.model, prune_ratio=0.5)
    
    def magnitude_prune(self, prune_ratio, mode="FULL"):
        print(f"!!!Applying {mode} magnitude pruning with ratio: {prune_ratio}")

        pruner = MagnitudePruner()
        pruner.apply_pruning(self.model, prune_ratio * 100, mode=mode)

    def generate_ws_graph(self, nodes, k=2, p=0.05):
        return nx.watts_strogatz_graph(nodes, k, p)

    def train(self, model, train_loader, epochs=30, lr=0.001):
        print("Training started...")
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

        epoch = 0 
        while self.running and int(epoch) < int(epochs):
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
            epoch += 1 

        print("Training complete.")
        self.message.emit("Training complete.")


    def dag_to_mlp_structure(self, G, input_size, output_size):
        num_nodes = len(G.nodes)
        layers = np.array_split(sorted(G.nodes), len(self.hidden_sizes))
        layer_sizes = [len(layer) for layer in layers]
        return [input_size] + layer_sizes + [output_size]
    
    def dag_to_lstm_structure(self, dag, input_size, output_size):
        node_layers = nx.get_node_attributes(dag, 'layer')
        max_layer = max(node_layers.values()) if node_layers else 0
        layer_sizes = [sum(1 for _ in filter(lambda x: x == l, node_layers.values())) for l in range(max_layer + 1)]

        return [input_size] + layer_sizes + [output_size]


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