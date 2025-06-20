import torch.nn as nn
import torch
import torch.nn.functional as F
import networkx as nx

class MLPNet(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            nn.init.constant_(linear.bias, 0.01)
            layers.append(linear)
            
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                layers.append(nn.Dropout(0.2))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_dim=10):
        super().__init__()
        
        self.lstms = nn.ModuleList()
        
        current_size = input_size
        for size in hidden_sizes:
            self.lstms.append(nn.LSTM(
                input_size=current_size,
                hidden_size=size,
                batch_first=True,
                dropout=0.0
            ))
            current_size = size
        
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(hidden_sizes[-1])
        self.fc = nn.Linear(hidden_sizes[-1], output_dim)
        for lstm in self.lstms:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
                    if 'bias_ih' in name:
                        param.data[size:2*size].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        h = x
        for lstm in self.lstms:
            h, _ = lstm(h)
            h = self.dropout(h)
        last_hidden = h[:, -1, :]
        normalized = self.batch_norm(last_hidden)
        output = self.fc(normalized)
        
        return output
    
class SparseMLPNet(nn.Module):
    def __init__(self, layer_sizes, hidden_sizes=None, dag_graph=None):
        super().__init__()
        if not isinstance(layer_sizes, list):
            layer_sizes = [layer_sizes[0]] + (hidden_sizes if hidden_sizes else []) + [layer_sizes[-1]]
        
        self.layer_sizes = layer_sizes
        self.dag_graph = dag_graph
        
        self.layers = nn.ModuleList()
        self.masks = nn.ParameterList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(len(self.layer_sizes) - 1):
            linear = nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
            nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            nn.init.constant_(linear.bias, 0.01)
            self.layers.append(linear)
            
            mask = nn.Parameter(torch.ones_like(linear.weight), requires_grad=False)
            self.masks.append(mask)
            
            if i < len(self.layer_sizes)-2:
                self.batch_norms.append(nn.BatchNorm1d(self.layer_sizes[i+1]))
    
        if dag_graph is not None:
            self._apply_graph_sparsity()
        
        self.dropout = nn.Dropout(0.2)
        self._apply_masks()
        self._scale_weights_for_sparsity()
    
    def _scale_weights_for_sparsity(self):
        with torch.no_grad():
            for layer_idx, (layer, mask) in enumerate(zip(self.layers, self.masks)):
                sparsity = 1.0 - (mask.sum() / mask.numel())
                if sparsity > 0.1:
                    scale_factor = torch.sqrt(1.0 / (1.0 - sparsity + 1e-6))
                    layer.weight.data *= scale_factor
    
    def _apply_graph_sparsity(self):
        if self.dag_graph is None:
            return
            
        node_layers = nx.get_node_attributes(self.dag_graph, 'layer')
        if not node_layers:
            return
        
        for layer_idx in range(len(self.masks)):
            mask = self.masks[layer_idx]
            
            # Find nodes in adjacent layers
            in_layer_nodes = [n for n, l in node_layers.items() if l == layer_idx]
            out_layer_nodes = [n for n, l in node_layers.items() if l == layer_idx + 1]
            
            if not in_layer_nodes or not out_layer_nodes:
                continue
                
            in_mapping = self._create_node_mapping(in_layer_nodes, mask.shape[1])
            out_mapping = self._create_node_mapping(out_layer_nodes, mask.shape[0])
            
            for in_idx, in_node in enumerate(in_layer_nodes):
                if in_idx >= len(in_mapping):
                    continue
                    
                for out_idx, out_node in enumerate(out_layer_nodes):
                    if out_idx >= len(out_mapping):
                        continue
                    
                    # Map
                    net_in_idx = in_mapping[in_idx]
                    net_out_idx = out_mapping[out_idx]
                    if net_in_idx >= mask.shape[1] or net_out_idx >= mask.shape[0]:
                        continue
                        
                    # Zero out weights where there's no edge
                    if not self.dag_graph.has_edge(in_node, out_node):
                        mask.data[net_out_idx, net_in_idx] = 0.0
    
    def _create_node_mapping(self, nodes, layer_size):
        mapping = {}
        if len(nodes) >= layer_size:
            for i in range(len(nodes)):
                mapping[i] = i % layer_size
        else:
            neurons_per_node = layer_size // len(nodes)
            remainder = layer_size % len(nodes)
            
            idx = 0
            for i in range(len(nodes)):
                extra = 1 if i < remainder else 0
                count = neurons_per_node + extra
                
                for j in range(count):
                    if idx < layer_size:
                        mapping[i] = idx
                        idx += 1
        
        return mapping
    
    def _apply_masks(self):
        with torch.no_grad():
            for layer, mask in zip(self.layers, self.masks):
                layer.weight.data *= mask
    
    def forward(self, x):
        h = x
        
        for i, (layer, mask) in enumerate(zip(self.layers, self.masks)):
            # Apply mask to maintain sparsity
            with torch.no_grad():
                layer.weight.data *= mask
            
            h = layer(h)
            # activation, batch norm, and dropout for all but the last layer
            if i < len(self.layers) - 1:
                h = F.relu(h)
                if i < len(self.batch_norms):
                    h = self.batch_norms[i](h)
                h = self.dropout(h)
        
        return h


class SparseLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_dim=10, dag_graph=None):
        super().__init__()
        
        if not isinstance(hidden_sizes, (list, tuple)):
            hidden_sizes = [hidden_sizes]
        
        self.dag_graph = dag_graph
        self.input_size = input_size
        self.output_dim = output_dim
        
        self.lstms = nn.ModuleList()
        current_input_size = input_size
        for hidden_size in hidden_sizes:
            self.lstms.append(nn.LSTM(
                input_size=current_input_size,
                hidden_size=hidden_size,
                batch_first=True
           ))
            current_input_size = hidden_size
        
        self.batch_norm = nn.BatchNorm1d(hidden_sizes[-1])
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_sizes[-1], output_dim)

        for lstm in self.lstms:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
                    # Set forget gate bias to 1.0 !!!important for LSTM!!!
                    if 'bias_ih' in name:
                        param.data[hidden_size:2*hidden_size].fill_(1.0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
        if dag_graph is not None:
            self._create_feature_masks()

    def _create_feature_masks(self):
        if self.dag_graph is None:
            return
        # Find input nodse
        input_nodes = [n for n in self.dag_graph.nodes() if str(n).startswith('input_') or str(n).startswith('layer0_')]
        
        if not input_nodes:
            return
        self.input_mask = torch.ones(self.input_size, device=next(self.parameters()).device)
        
        for i in range(self.input_size):
            node_name = f'input_{i}'
            if node_name in self.dag_graph.nodes() and self.dag_graph.out_degree(node_name) == 0:
                self.input_mask[i] = 0.0

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), -1, self.input_size)
        
        if hasattr(self, 'input_mask'):
            expanded_mask = self.input_mask.expand(x.size(0), x.size(1), -1)
            x = x * expanded_mask
        
        h = x
        for i, lstm in enumerate(self.lstms):
            h, _ = lstm(h)
            if i < len(self.lstms) - 1:
                h = self.dropout(h)
        final_hidden = h[:, -1, :]
        normalized = self.batch_norm(final_hidden)
        dropped = self.dropout(normalized)
        output = self.fc(dropped)
        
        return output
