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
from time import time

class Trainer(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)


    def __init__(self, model_type, dataset_type, lr=None, hidden_sizes=[6,6,6], loss="CrossEntropy", optimizer="Adam", epochs=30, k=2, p=0.05,graph_type="Full", N=250, batch_size=64, layer_count=5, index=None):

        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.model_type = model_type
        self.model = None
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.k = k
        self.p = p
        self.dataset_type = dataset_type
        self.graph_type = graph_type
        self.batch_size = batch_size
        self.layer_count = layer_count
        self.index = index
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if loss == "CrossEntropy":
            self.criterion = nn.CrossEntropyLoss()
        else: #for now default to MSELoss
            self.criterion = nn.MSELoss()
        self.running = True
        self.current_epoch = 0
        self.training_metrics = {
            'final_train_loss': None,
            'final_train_accuracy': None,
            'final_val_loss': None,
            'final_val_accuracy': None,
            'model_metrics': {},
            'graph_metrics': {}
        }
        self.pruning_history = []

        
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
        

        if self.dataset_type == "MNIST":
            self.feature_size = 28
            self.sequence_length = 28
        else:  # CIFAR
            self.feature_size = 32 * 3
            self.sequence_length = 32

        # Load datasets
        train_dataset = dataset_config["dataset"](
            root="./data", train=True, transform=transform, download=True)
        test_dataset = dataset_config["dataset"](
            root="./data", train=False, transform=transform, download=True)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.dag_graph = None
        if self.graph_type == "WS":
            self.dag_graph = self.generate_ws_dag(nodes=self.hidden_sizes, k=self.k, p=self.p, target_layers=self.layer_count )
            if self.model_type == "MLP":
                mlp_structure = self.dag_to_mlp_structure(self.dag_graph, input_size, num_classes)
                self.model = SparseMLPNet(mlp_structure, hidden_sizes=self.hidden_sizes).to(self.device)
            else:  # LSTM
                lstm_structure = self.dag_to_lstm_structure(self.dag_graph)
                self.hidden_sizes = lstm_structure[1:-1] if len(lstm_structure) > 2 else lstm_structure
                self.model = SparseLSTMNet(input_size=self.feature_size, 
                                         hidden_sizes=self.hidden_sizes, 
                                         output_dim=num_classes).to(self.device)

        elif self.graph_type == "Full":
            self.dag_graph = self.generate_fully_connected_graph(sum(self.hidden_sizes))
            self.dag_graph = self.dag_graph
            
            if self.model_type == "MLP":
                mlp_structure = self.dag_to_mlp_structure(self.dag_graph, input_size, num_classes)
                self.model = MLPNet(mlp_structure).to(self.device)
            else:  # LSTM
                self.model = LSTMNet(input_size=self.feature_size, 
                                   hidden_sizes=self.hidden_sizes, 
                                   output_dim=num_classes).to(self.device)

        elif self.graph_type == "BA":
            self.dag_graph = self.generate_ba_dag(nodes=sum(self.hidden_sizes), 
                                           edges_per_node=self.k, 
                                           target_layers=self.layer_count)
            if self.model_type == "MLP":
                mlp_structure = self.dag_to_mlp_structure(self.dag_graph, input_size, num_classes)
                self.model = SparseMLPNet(mlp_structure).to(self.device)
            else:  # LSTM
                self.model = SparseLSTMNet(input_size=self.feature_size, 
                                         hidden_sizes=self.hidden_sizes, 
                                         output_dim=num_classes).to(self.device)
        
        if self.graph_type == "WS" or self.graph_type == "BA":
            print(f"Parameters of the trainer: {self.epochs} epoch, default learning_rate, {self.optimizer} optimizer, {self.criterion} loss function, {self.k} k, {self.p} p, {self.dataset_type} dataset, {self.hidden_sizes} hidden sizes")
            progress_message = f"Parameters of the trainer: {self.epochs} epoch, {self.lr} learning_rate, {self.optimizer} optimizer, {self.criterion} loss function, {self.k} k, {self.p} p, {self.dataset_type} dataset, {self.hidden_sizes} hidden sizes"
        else:
            print(f"Parameters of the trainer: {self.epochs} epoch, default learning_rate, {self.optimizer} optimizer, {self.criterion} loss function, {self.dataset_type} dataset, {self.hidden_sizes} hidden sizes")
            progress_message = f"Parameters of the trainer: {self.epochs} epoch, {self.optimizer} optimizer, {self.criterion} loss function, {self.dataset_type} dataset, {self.hidden_sizes} hidden sizes"
        self.message.emit(progress_message)



    def run(self):
        print("Starting the training process...")
        progress_message = "Starting the training process..."
        self.message.emit(progress_message)

        self.train(self.model, self.train_loader, self.epochs - self.current_epoch, lr=self.lr)
        self.finished.emit()  # notify gui when done
    
    def stop(self):
        self.running = False
        self.terminate()
        self.wait()
        self.finished.emit()

    
    def magnitude_prune(self, prune_ratio, mode="FULL"):
        try:
            if not hasattr(self, 'model'):
                raise AttributeError("Model attribute missing - was __init__() called?")
                
            if isinstance(self.model, str):
                raise ValueError(f"Model is still a string ('{self.model}') - call load_data_and_create_graph() first")
                
            if self.model is None:
                raise ValueError("Model is None - initialization failed in load_data_and_create_graph()")
                
            try:
                first_param = next(self.model.parameters(), None)
                if first_param is None:
                    raise RuntimeError("Model exists but has no parameters")
            except Exception as e:
                raise RuntimeError(f"Parameter access failed: {str(e)}") from e
                
            print("\n=== Pre-Pruning Validation ===")
            print(f"Model type: {type(self.model).__name__}")
            print(f"Device: {next(self.model.parameters()).device}")
            print(f"Parameter tensors: {sum(1 for _ in self.model.parameters())}")
            
            print(f"\nApplying {mode} pruning at {prune_ratio:.0%} ratio")
            pruner = MagnitudePruner()
            pruner.apply_pruning(self.model, prune_ratio * 100, mode=mode)
            
            self.record_pruning_metrics(prune_type=mode, prune_percent=prune_ratio)
            print("\n=== Pruning Results ===")
            total_params = sum(p.numel() for p in self.model.parameters())
            zero_params = sum((p == 0).sum().item() for p in self.model.parameters())
            actual_sparsity = zero_params / total_params if total_params > 0 else 0
            print(f"Actual sparsity: {actual_sparsity:.2%} (target: {prune_ratio:.2%})")
            
            self.training_metrics['model_metrics'] = {
                'total_parameters': total_params,
                'global_sparsity': actual_sparsity,
                'prune_mode': mode
            }


            return True
            
        except Exception as e:
            error_msg = f"Pruning failed at step {len(self.pruning_history)+1}: {str(e)}"
            print(error_msg)
            self.message.emit(error_msg)
            return False



    def train(self, model, train_loader, epochs=30, lr=0.001):
        print("Training started...")
        model.to(self.device)
        
        # Initialize optimizer
        optimizers = {
            "Adam": optim.Adam,
            "SGD": optim.SGD,
            "RMSprop": optim.RMSprop,
            "Adadelta": optim.Adadelta,
            "Adagrad": optim.Adagrad
        }
        self.optimizer = optimizers.get(self.optimizer, optim.Adam)(model.parameters(), lr=lr)
        epoch = 0 

        for epoch in range(epochs):
            if not self.running:
                print("Training stopped early")
                return

            model.train()
            total_loss, correct, total = 0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Handle different input formats
                if isinstance(model, (LSTMNet, SparseLSTMNet)):
                    images = images.view(images.size(0), self.sequence_length, self.feature_size)
                else:
                    images = images.view(images.size(0), -1)

                self.optimizer.zero_grad()
                outputs = model(images)
                
                if isinstance(self.criterion, nn.MSELoss):
                    labels_one_hot = torch.zeros(labels.size(0), 10).to(self.device)
                    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
                    loss = self.criterion(outputs, labels_one_hot)
                else:
                    loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # Update metrics
            self.current_epoch = epoch
            train_loss = total_loss / len(train_loader)
            train_acc = 100. * correct / total
            val_loss, val_acc = self.validate(model)
            
            self.training_metrics.update({
                'final_train_loss': train_loss,
                'final_train_accuracy': train_acc,
                'final_val_loss': val_loss,
                'final_val_accuracy': val_acc
            })

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            self.message.emit(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")


        print("Training complete.")
        self.message.emit("Training complete.")

    def validate(self, model=None):
        model = model or self.model
        model.eval()
        

        total_loss, correct, total = 0, 0, 0
        
        if not hasattr(self, 'test_loader'):
            return float('nan'), 0.0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                if isinstance(model, (LSTMNet, SparseLSTMNet)):
                    if hasattr(self, 'sequence_length') and hasattr(self, 'feature_size'):
                        images = images.view(images.size(0), self.sequence_length, self.feature_size)
                    else:
                        images = images.view(images.size(0), -1)
                else:
                    images = images.view(images.size(0), -1)
                
                outputs = model(images)
                
                if isinstance(self.criterion, nn.MSELoss):
                    labels_one_hot = torch.zeros(labels.size(0), 10).to(self.device)
                    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
                    loss = self.criterion(outputs, labels_one_hot)
                else:
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        if len(self.test_loader) == 0:
            return float('nan'), 0.0
        
        return total_loss / len(self.test_loader), 100. * correct / total

    def get_state(self):
        self.graph_metrics = self.calculate_graph_metrics(self.dag_graph) if hasattr(self, 'dag_graph') else {}
        return {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self.optimizer, 'state_dict') else None,
            'current_epoch': self.current_epoch,
            'index': self.index,
            'training_metrics': self.training_metrics,
            'hidden_sizes': self.hidden_sizes,
            'lr': self.lr,
            'model_type': self.model_type,
            'dataset_type': self.dataset_type,
            'graph_type': self.graph_type,
            'k': self.k,
            'p': self.p,
            'layer_count': self.layer_count,
            'batch_size': self.batch_size,
            'pruning_history': self.pruning_history,
            'dag_graph': nx.node_link_data(self.dag_graph) if hasattr(self, 'dag_graph') else None,
            'model': self.model.__class__.__name__ if hasattr(self, 'model') else None,
            'optimizer': self.optimizer.__class__.__name__ if hasattr(self, 'optimizer') else None,
            'criterion': self.criterion.__class__.__name__ if hasattr(self, 'criterion') else None,
            'training_metrics': self.training_metrics if hasattr(self, 'training_metrics') else None,
            'graph_metrics': self.graph_metrics if hasattr(self, 'graph_metrics') else None,
            'prune_type' : self.pruning_history[-1]['type'] if self.pruning_history else None,
        }


    
    def set_state(self, state):
        self.current_epoch = state.get('current_epoch', 0)
        self.index = state.get('index')
        self.training_metrics = state.get('training_metrics', {})
        self.hidden_sizes = state.get('hidden_sizes', [])
        self.lr = state.get('lr')
        self.model_type = state.get('model_type')
        self.dataset_type = state.get('dataset_type')
        self.graph_type = state.get('graph_type')
        self.k = state.get('k', 2)
        self.p = state.get('p', 0.05)
        self.layer_count = state.get('layer_count', 5)
        self.batch_size = state.get('batch_size', 64)
        self.pruning_history = state.get('pruning_history', [])

        self.load_data_and_create_graph()
        if self.model and state['model_state_dict']:
            self.model.load_state_dict(state['model_state_dict'])
        if self.model:
            optimizers = {
                "Adam": optim.Adam,
                "SGD": optim.SGD,
                "RMSprop": optim.RMSprop,
                "Adadelta": optim.Adadelta,
                "Adagrad": optim.Adagrad
            }
            opt_class = optimizers.get(self.optimizer, optim.Adam)
            self.optimizer = opt_class(self.model.parameters(), lr=self.lr or 0.001)
            if state['optimizer_state_dict']:
                self.optimizer.load_state_dict(state['optimizer_state_dict'])

        if state.get('dag_graph'):
            self.dag_graph = nx.node_link_graph(state['dag_graph'])
        if state.get('graph_metrics'):
            self.graph_metrics = state['graph_metrics']
        if state.get('model'):
            model_class = globals().get(state['model'])
            if model_class:
                self.model = model_class(self.hidden_sizes).to(self.device)
            else:
                raise ValueError(f"Model class {state['model']} not found")
        if state.get('optimizer'):
            optimizer_class = globals().get(state['optimizer'])
            if optimizer_class:
                self.optimizer = optimizer_class(self.model.parameters(), lr=self.lr or 0.001)
            else:
                raise ValueError(f"Optimizer class {state['optimizer']} not found")
        if state.get('criterion'):
            criterion_class = globals().get(state['criterion'])
            if criterion_class:
                self.criterion = criterion_class()
            else:
                raise ValueError(f"Criterion class {state['criterion']} not found")
        if state.get('training_metrics'):
            self.training_metrics = state['training_metrics']
        if state.get('graph_metrics'):
            self.graph_metrics = state['graph_metrics']
        if state.get('prune_type'):
            self.pruning_history.append({'type': state['prune_type']})

    def save_model(self, path, neural_networks):
        state = self.get_state()
        state['neural_networks'] = neural_networks
        torch.save(state, path)
        print(f"Saved model and trainer state to {path}")

    def load_model(self, path, neural_networks):
        state = torch.load(path, map_location=self.device)
        self.set_state(state)
        self.index = state.get('index')
        print(f"Model and trainer state loaded from {path}")
        return state.get('neural_networks', neural_networks), self.index


    def calculate_graph_metrics(self, G):
        if not G:
            return {}
        if self.model.named_parameters() is None:
            return {}
        metrics = {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'global_sparsity': 0.0,
            'layer_sparsity': {},
            'edge_betweenness': [],
            'closeness': [],
            'eccentricity': [],
            'degree': [],
            'betweenness': []
        }
        
        total_weights = 0
        zero_weights = 0
        
        for name, param in self.model.named_parameters():
            metrics['total_parameters'] += param.numel()
            if param.requires_grad:
                metrics['trainable_parameters'] += param.numel()
            
            if 'weight' in name:
                #sparsity
                zeros = (param == 0).sum().item()
                total = param.numel()
                metrics['layer_sparsity'][name] = zeros / total
                zero_weights += zeros
                total_weights += total
        
        if total_weights > 0:
            metrics['global_sparsity'] = zero_weights / total_weights

        try:
            metrics['edge_betweenness'] = nx.edge_betweenness_centrality(G)
            metrics['node_betweenness'] = nx.betweenness_centrality(G)
            metrics['closeness'] = nx.closeness_centrality(G)
            metrics['degree'] = dict(G.degree())

            if nx.is_strongly_connected(G):
                metrics['eccentricity'] = nx.eccentricity(G)
            else:
                # Use the largest strongly connected component ?
                largest_scc = max(nx.strongly_connected_components(G), key=len)
                subgraph = G.subgraph(largest_scc)
                metrics['eccentricity'] = nx.eccentricity(subgraph)

        except Exception as e:
            self.message.emit(f"Graph metric calculation error: {e}")

        return metrics
    

    def calculate_model_metrics(self):
        if isinstance(self.model, str):
            return {
                'total_parameters': 0,
                'trainable_parameters': 0,
                'global_sparsity': 0.0,
                'layer_sparsity': {}
            }
        metrics = {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'global_sparsity': 0.0,
            'layer_sparsity': {}
        }
        
        total_weights = 0
        zero_weights = 0
        
        for name, param in self.model.named_parameters():
            metrics['total_parameters'] += param.numel()
            if param.requires_grad:
                metrics['trainable_parameters'] += param.numel()
            
            if 'weight' in name:
                zeros = (param == 0).sum().item()
                total = param.numel()
                metrics['layer_sparsity'][name] = zeros / total
                zero_weights += zeros
                total_weights += total
        
        if total_weights > 0:
            metrics['global_sparsity'] = zero_weights / total_weights
        
        return metrics
    
    def record_pruning_metrics(self, prune_type, prune_percent):
        try:
            val_loss, val_acc = self.validate()
        except Exception as e:
            print(f"Validation failed during pruning recording: {e}")
            val_loss, val_acc = float('nan'), 0.0
        
        layer_sparsity = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                try:
                    layer_name = name.split('.')[0]
                    zero_count = (param == 0).sum().item()
                    layer_sparsity[layer_name] = zero_count / param.numel()
                except Exception as e:
                    print(f"Failed to calculate sparsity for {name}: {e}")
        
        try:
            graph_metrics = self.calculate_graph_metrics(self.dag_graph) if hasattr(self, 'dag_graph') else {}
        except Exception as e:
            print(f"Failed to calculate graph metrics: {e}")
            graph_metrics = {}
        
        self.pruning_history.append({
            'step': len(self.pruning_history),
            'type': prune_type,
            'global_percent': prune_percent,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'layer_sparsity': layer_sparsity,
            'graph_metrics': graph_metrics
        })


    def calculate_model_metrics(self):
        metrics = {
            'layer_parameters': {}, 
            'layer_sparsity': {}
        }
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                layer_name = name.split('.')[0]
                metrics['layer_parameters'][layer_name] = param.numel()
                if isinstance(self.model, (SparseMLPNet, SparseLSTMNet)):
                    metrics['layer_sparsity'][layer_name] = (param == 0).sum().item() / param.numel()
        
        return metrics
    
    def generate_fully_connected_graph(self, nodes):
        G = nx.complete_graph(nodes, create_using=nx.DiGraph)
        for u,v in list(G.edges):
            if u > v:
                G.remove_edge(u, v)
        
        layers = {node: node for node in G.nodes()}
        nx.set_node_attributes(G, layers, 'layer')

        return G

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
            count = sum(1 for layer in node_layers.values() if layer == l)
            layer_sizes.append(count)

        if input_size is not None and output_size is not None:
            return [input_size] + layer_sizes + [output_size]
        return layer_sizes
    

    def print_summary(self): #just for testing/debug
        print("\n=== Model Summary ===")
        print(f"Model Type: {getattr(self.model, '__class__', type(None)).__name__}")
        
        model_metrics = self.training_metrics.get('model_metrics', {})
        
        total_params = model_metrics.get('total_parameters')
        if isinstance(total_params, (int, float)):
            print(f"Total Parameters: {total_params:,}")
        else:
            print("Total Parameters: None")
        
        trainable_params = model_metrics.get('trainable_parameters', total_params) 
        if isinstance(trainable_params, (int, float)):
            print(f"Trainable Parameters: {trainable_params:,}")
        else:
            print("Trainable Parameters: None")
        
        sparsity = model_metrics.get('global_sparsity')
        if isinstance(sparsity, (int, float)):
            print(f"Global Sparsity: {sparsity:.2%}")
        else:
            print("Global Sparsity: None")
        
        print("\n=== Training Performance ===")
        train_acc = self.training_metrics.get('final_train_accuracy')
        val_acc = self.training_metrics.get('final_val_accuracy')
        
        print(f"Final Train Accuracy: {train_acc:.2f}%" if isinstance(train_acc, (int, float)) else "Final Train Accuracy: None")
        print(f"Final Val Accuracy: {val_acc:.2f}%" if isinstance(val_acc, (int, float)) else "Final Val Accuracy: None")
        
        if self.pruning_history:
            print("\n=== Pruning Summary ===")
            print(f"Pruning Steps: {len(self.pruning_history)}")
            last_prune = self.pruning_history[-1]
            
            prune_type = last_prune.get('type', 'N/A')
            prune_percent = last_prune.get('global_percent')
            prune_acc = last_prune.get('val_accuracy')
            
            print(f"Last Pruning: {prune_type} at {prune_percent:.0%}" if isinstance(prune_percent, (int, float)) 
                else f"Last Pruning: {prune_type}")
            print(f"Accuracy After Pruning: {prune_acc:.2f}%" if isinstance(prune_acc, (int, float)) 
                else "Accuracy After Pruning: Not available")
        
        if hasattr(self, 'graph_metrics') and self.graph_metrics:
            print("\n=== Graph Metrics ===")
            print(f"Nodes: {len(self.graph_metrics.get('degree_centrality', {}))}")
            
            edge_counts = self.graph_metrics.get('edge_counts', {})
            total_edges = sum(edge_counts.values()) if edge_counts else 0
            print(f"Edges: {total_edges}")
            
            degree_values = list(self.graph_metrics.get('degree_centrality', {}).values())
            if degree_values:
                print(f"Average Degree: {np.mean(degree_values):.4f}")
            
            betweenness_values = list(self.graph_metrics.get('node_betweenness', {}).values())
            if betweenness_values:
                print(f"Average Betweenness: {np.mean(betweenness_values):.4f}")