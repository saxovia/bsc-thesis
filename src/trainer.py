from src.neuralnetwork import MLPNet, LSTMNet, SparseMLPNet, SparseLSTMNet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PyQt6.QtCore import QThread, pyqtSignal
from src.pruner import MagnitudePruner, RandomPruner, L1Pruner
from collections import defaultdict

# Define the Trainer class


class Trainer(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)


    def __init__(self, model_type, dataset_type, lr=None, hidden_sizes=[6,6,6], loss="CrossEntropy", optimizer="Adam", epochs=30, k=2, p=0.05,graph_type="Full", N=250, batch_size=64, layer_count=5, index=None):

        super().__init__()
        self.hidden_sizes = hidden_sizes
        #if model_type == "MLP":
        #    self.model = MLPNet(hidden_sizes)
        #elif model_type == "LSTM":
        #    self.model = LSTMNet(hidden_sizes)
        self.model = model_type
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.k = k
        self.p = p
        self.dataset_type = dataset_type
        self.graph_type = graph_type
        #self.N = int(N)
        self.batch_size = batch_size
        self.layer_count = layer_count
        self.index = index

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
        if self.dataset_type == "MNIST":
            channels = 1
        else:  # CIFAR-10 or CIFAR-100
            channels = 3

        dataset_config = dataset_info[self.dataset_type]
        input_size_flatten = dataset_config["input_size"]
        channels = 1 if self.dataset_type == "MNIST" else 3
        sequence_length = int((input_size_flatten / channels) ** 0.5)
        feature_size = sequence_length * channels
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        if self.dataset_type == "MNIST":
            self.feature_size = 28  # For MNIST: 28 features (28x1)
        elif self.dataset_type.startswith("CIFAR"):
            self.feature_size = 32*3  # For CIFAR: 96 features (32x3)
        input_size = dataset_config["input_size"]
        num_classes = dataset_config["num_classes"]
        transform = dataset_config["transform"]
        default_batch_size = dataset_config["default_batch_size"]
        batch_size = self.batch_size if self.batch_size is not None else default_batch_size #? TODO

        train_dataset = dataset_config["dataset"](root="./data", train=True, transform=transform, download=True)
        test_dataset = dataset_config["dataset"](root="./data", train=False, transform=transform, download=True)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


        print(f"Loaded {self.dataset_type} dataset.")
        progress_message = f"Loaded {self.dataset_type} dataset."
        self.message.emit(progress_message)

        dag_graph = None
        if self.graph_type == "WS":
            #ws_graph = self.generate_ws_graph(self.hidden_sizes, k=self.k, p=self.p)
            #dag_graph = self.ws_to_dag(ws_graph)
            dag_graph = self.generate_ws_dag(nodes=self.hidden_sizes, k=self.k, p=self.p, target_layers=self.layer_count )
            if self.model == "MLP":
                mlp_structure = self.dag_to_mlp_structure(dag_graph, input_size, num_classes)
                self.model = SparseMLPNet(mlp_structure, hidden_sizes=self.hidden_sizes).to(self.device)
                print("MLP NN created.")
                self.message.emit("MLP NN created.")
            elif self.model == "LSTM":
                lstm_structure = self.dag_to_lstm_structure(dag_graph)
                self.model = SparseLSTMNet(input_size=self.feature_size, hidden_sizes=self.hidden_sizes, output_dim=num_classes).to(self.device)
                print("LSTM NN created.")
                self.message.emit("LSTM NN created.")


        elif self.graph_type == "Full":
            if self.model == "MLP":
                dag_graph = self.generate_fully_connected_graph(sum(self.hidden_sizes))
                mlp_structure = self.dag_to_mlp_structure(dag_graph, input_size, num_classes)
                self.model = MLPNet(mlp_structure).to(self.device)
                print("MLP NN created.")
                self.message.emit("MLP NN created.")
            elif self.model == "LSTM":
                dag_graph = self.generate_fully_connected_graph(sum(self.hidden_sizes))
                lstm_structure = self.dag_to_lstm_structure(dag_graph, input_size, num_classes)
                self.model = LSTMNet(input_size=self.feature_size, hidden_sizes=self.hidden_sizes, output_dim=num_classes).to(self.device)
                print("LSTM NN created.")
                self.message.emit("LSTM NN created.")

        elif self.graph_type == "BA":
            dag_graph = self.generate_ba_dag(nodes=self.hidden_sizes, edges_per_node=self.k, target_layers=self.layer_count)
            if self.model == "MLP":
                mlp_structure = self.dag_to_mlp_structure(dag_graph, input_size, num_classes)
                self.model = SparseMLPNet(mlp_structure).to(self.device)
                print("MLP NN created.")
                self.message.emit("MLP NN created.")
            elif self.model == "LSTM":
                lstm_structure = self.dag_to_lstm_structure(dag_graph)
                self.model = SparseLSTMNet(input_size=self.feature_size, hidden_sizes=self.hidden_sizes, output_dim=num_classes).to(self.device)
                print("LSTM NN created.")
                self.message.emit("LSTM NN created.")
        


        print(f"Parameters of the trainer: {self.epochs} epoch, default learning_rate, {self.optimizer} optimizer, {self.criterion} loss function, {self.k} k, {self.p} p, {self.dataset_type} dataset, {self.hidden_sizes} hidden sizes")
        progress_message = f"Parameters of the trainer: {self.epochs} epoch, {self.lr} learning_rate, {self.optimizer} optimizer, {self.criterion} loss function, {self.k} k, {self.p} p, {self.dataset_type} dataset, {self.hidden_sizes} hidden sizes"
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
        G = nx.complete_graph(nodes, create_using=nx.DiGraph)
        for u,v in list(G.edges):
            if u > v:
                G.remove_edge(u, v)
        
        layers = {node: node for node in G.nodes()}
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
        print(f"Applying {mode} magnitude pruning with ratio: {prune_ratio}")
        self.message.emit(f"Applying {mode} magnitude pruning with ratio: {prune_ratio}")
        pruner = MagnitudePruner()
        pruner.apply_pruning(self.model, prune_ratio * 100, mode=mode)


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
                if not self.running:  # Exit immediately if stop() is called
                    print("Stopping training early...")
                    return  

                images, labels = images.to(self.device), labels.to(self.device)
                if isinstance(model, (LSTMNet, SparseLSTMNet)):
                    if self.dataset_type == "MNIST":
                        # For MNIST: (batch_size, 28, 28)
                        images = images.view(images.size(0), 28, 28)
                    elif self.dataset_type.startswith("CIFAR"):
                        # For CIFAR: (batch_size, 32, 32*3) since CIFAR has 32x32 images with 3 channels
                        images = images.view(images.size(0), 32, 32*3)
                else:
                    images = images.view(images.size(0), -1)

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

    def generate_ws_graph(self, nodes, k=2, p=0.05):
        return nx.watts_strogatz_graph(nodes, k, p)
    def generate_ws_dag(self, nodes, k=2, p=0.7, target_layers=5):
        # Try to generate a Ws graph until it is connected
        while True:
            ws = nx.watts_strogatz_graph(nodes, k, p)
            if nx.is_connected(ws):
                break
        #DAG
        dag = nx.DiGraph()
        dag.add_nodes_from(range(nodes))
        for u, v in ws.edges():
            if u < v: #lower triangular part of the graph
                dag.add_edge(u, v)
        
        # Getbalanced distribution
        """     
        layers = {}
        for i, node in enumerate(nx.topological_sort(dag)):
            layers[node] = i // (len(dag.nodes) // target_layers)
        return layers
        """
        nodes_per_layer = nodes // target_layers
        layers = {}
        for i, node in enumerate(dag.nodes()):
            layers[node] = min(i // nodes_per_layer, target_layers - 1)
        
        # This is here so that edges only go forward
        for u, v in list(dag.edges()):
            if layers[u] >= layers[v]:
                dag.remove_edge(u, v)
        
        nx.set_node_attributes(dag, layers, 'layer')
        return dag
    def generate_ba_dag(self, nodes, edges_per_node, target_layers=5):
        ba_graph = nx.barabasi_albert_graph(nodes, edges_per_node)
        dag = nx.DiGraph()
        dag.add_nodes_from(ba_graph.nodes)
        for u, v in ba_graph.edges():
            if u < v:
                dag.add_edge(u, v)

        nodes_per_layer = nodes // target_layers
        layers = {}
        for i, node in enumerate(dag.nodes()):
            layers[node] = min(i // nodes_per_layer, target_layers - 1)

        for u, v in list(dag.edges()):
            if layers[u] >= layers[v]:
                dag.remove_edge(u, v)

        nx.set_node_attributes(dag, layers, 'layer')
        return dag

    def dag_to_mlp_structure(self, G, input_size, output_size):
        if isinstance(self.hidden_sizes, list) and len(self.hidden_sizes) > 0:
            return [input_size] + self.hidden_sizes + [output_size]
        else:
            layers = nx.get_node_attributes(G, 'layer')
            if not layers:
                return [input_size, output_size]
            layer_counts = defaultdict(int)
            for node, layer in layers.items():
                layer_counts[layer] += 1
            layer_sizes = [layer_counts[l] for l in sorted(layer_counts)]
            
            if layer_sizes:
                layer_sizes[0] = input_size
                layer_sizes[-1] = output_size
            
            return layer_sizes
        
    def dag_to_lstm_structure(self, dag, input_size=None, output_size=None):
        if self.graph_type == "Full":
            return self.hidden_sizes
        
        node_layers = nx.get_node_attributes(dag, 'layer')
        max_layer = max(node_layers.values()) if node_layers else 0
        layer_sizes = []

        for l in range(max_layer + 1):
            count = 0
            for layer in node_layers.values():
                if layer==l:
                    count+=1
            layer_sizes.append(count)

        if input_size is not None and output_size is not None:
            return [input_size] + layer_sizes + [output_size]
        else:
            return layer_sizes



    def get_layer_adjacency(self, dag):
        layer_adj = defaultdict(list)
        for u, v in dag.edges():
            u_layer = dag.nodes[u]['layer']
            v_layer = dag.nodes[v]['layer']
            if u_layer < v_layer and u_layer not in layer_adj[v_layer]:
                layer_adj[v_layer].append(u_layer)
        max_layer = max(layer_adj.keys(), default=-1)
        layer_connections = {}
        for layer in range(max_layer + 1):
            predecessors = sorted(layer_adj.get(layer, []))
            layer_connections[layer] = predecessors
        return layer_connections
    

    #UNUSED FUNCTION
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