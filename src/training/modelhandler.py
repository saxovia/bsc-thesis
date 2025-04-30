from src.training.neuralnetwork import MLPNet, LSTMNet, SparseMLPNet, SparseLSTMNet
import networkx as nx
from collections import defaultdict
import torch
import io
import pickle

class ModelHandler:
    def __init__(self, model_type, hidden_sizes, device):
    
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.device = device
        self.model = None

    def create_model(self, graph_type, dag_graph, input_size, output_size, feature_size=None):

        if graph_type == "WS" or graph_type == "BA":
            if self.model_type == "MLP":
                mlp_structure = self.dag_to_mlp_structure(dag_graph, input_size, output_size)
                self.model = SparseMLPNet(mlp_structure, hidden_sizes=self.hidden_sizes, dag_graph=dag_graph).to(self.device)
            else:  # LSTM
                lstm_structure = self.dag_to_lstm_structure(dag_graph, input_size, output_size)
                self.hidden_sizes = lstm_structure[1:-1] if len(lstm_structure) > 2 else lstm_structure
                self.model = SparseLSTMNet(input_size=feature_size, 
                                           hidden_sizes=self.hidden_sizes, 
                                           output_dim=output_size, dag_graph=dag_graph).to(self.device)
        elif graph_type == "Full":
            if self.model_type == "MLP":
                mlp_structure = self.dag_to_mlp_structure(dag_graph, input_size, output_size)
                self.model = MLPNet(mlp_structure).to(self.device)
            else:  # LSTM
                self.model = LSTMNet(input_size=feature_size, 
                                     hidden_sizes=self.hidden_sizes, 
                                     output_dim=output_size).to(self.device)
        return self.model

    def calculate_metrics(self):
        if not self.model:
            return {
                'total_parameters':0,
                'trainable_parameters':0,
                'global_sparsity':0.0,
                'layer_sparsity': {}
            }

        metrics = {
            'total_parameters': 0,
            'trainable_parameters':0,
            'global_sparsity': 0.0,
            'layer_sparsity': {}
        }

        total_weights = 0
        zero_weights = 0

        for name, param in self.model.named_parameters():
            metrics['total_parameters'] += param.numel()
            if param.requires_grad:
                metrics['trainable_parameters'] += param.numel()


            if 'weight' in name and isinstance(param, torch.Tensor):
                zeros = (param==0).sum().item()
                total = param.numel()
                metrics['layer_sparsity'][name] = zeros/total
                zero_weights += zeros
                total_weights += total
        if total_weights > 0:
            metrics['global_sparsity'] = zero_weights/total_weights

        return metrics

    def dag_to_mlp_structure(self, dag_graph, input_size, output_size):
        # Check if the graph has node attributes for layers
        layers = nx.get_node_attributes(dag_graph, 'layer')
        if not layers:
            return [input_size, output_size]

        # Create a dictionary to count the number of nodes in each layer
        layer_counts = defaultdict(int)
        for node, layer in layers.items():
            layer_counts[layer] += 1

        # Sort the layers and create a list of layer sizes
        layer_sizes = [layer_counts[l] for l in sorted(layer_counts)]
        layer_sizes[0] = input_size
        layer_sizes[-1] = output_size

        return layer_sizes

    def dag_to_lstm_structure(self, dag_graph, input_size=None, output_size=None):
        node_layers = nx.get_node_attributes(dag_graph, 'layer')
        # Check if the graph has node attributes for layers
        # If not, return an empty list or a list with input and output sizes depending on the parameters
        if not node_layers:
            if input_size is not None and output_size is not None:
                return [input_size, output_size]
            return []
        # Create a dictionary to count the number of nodes in each layer
        max_layer = max(node_layers.values())
        # Create a list to store the sizes of each layer
        layer_sizes = [0] * (max_layer+1)

        # Count the number of nodes in each layer
        for layer in node_layers.values():
            layer_sizes[layer] += 1
        

        # If input_size and output_size are given, adjust the first and last layers accordingly
        if input_size is None and output_size is None:
            return layer_sizes
        if input_size is not None and output_size is None:
            return [input_size] + layer_sizes 
        if input_size is not None and output_size is not None:
            return [input_size] + layer_sizes + [output_size]
        
        return layer_sizes
    
    def serialize_graph(self, dag_graph):
        buffer = io.BytesIO()
        pickle.dump(dag_graph, buffer)
        buffer.seek(0)
        return buffer.getvalue()
