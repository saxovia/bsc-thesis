from collections import defaultdict
import networkx as nx
import random

class GraphHandler:
    def __init__(self):
        pass

    def create_dag_graph(self, nodes, graph_type, k=2, p=0.05, target_layers=5, layer_count=5):
        if graph_type == "Full":
            return self.generate_fully_connected_graph(nodes)
        elif graph_type == "WS":
            return self.generate_ws_dag(nodes, k=k, p=p)
        elif graph_type == "BA":
            return self.generate_ba_dag(nodes, k=k)
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")

    def generate_fully_connected_graph(self, nodes, first_layer_size=10):
        if isinstance(nodes, list):
            total_nodes = sum(nodes)
            G = nx.DiGraph()
            G.add_nodes_from(range(total_nodes))
            
            layers = {}
            current_index = 0
            for layer_idx, count in enumerate(nodes):
                for i in range(count):
                    layers[current_index] = layer_idx
                    current_index += 1

            # Fully connect layer i to layer i+1
            for i in range(len(nodes) - 1):
                layer_i_nodes = [n for n in layers if layers[n] == i]
                layer_j_nodes = [n for n in layers if layers[n] == i + 1]
                for u in layer_i_nodes:
                    for v in layer_j_nodes:
                        G.add_edge(u, v)

        elif isinstance(nodes, int):
            G = nx.DiGraph()
            G.add_nodes_from(range(nodes))
            
            layers = {}
            nodes_per_layer = max(1, nodes // 5)
            current_layer = 0
            remaining_nodes = nodes
            
            while remaining_nodes > 0:
                layer_size = min(nodes_per_layer, remaining_nodes)
                for i in range(layer_size):
                    node_idx = nodes - remaining_nodes + i
                    layers[node_idx] = current_layer
                remaining_nodes -= layer_size
                current_layer += 1

            for i in range(current_layer - 1):
                layer_i_nodes = [n for n in layers if layers[n] == i]
                layer_j_nodes = [n for n in layers if layers[n] == i + 1]
                for u in layer_i_nodes:
                    for v in layer_j_nodes:
                        G.add_edge(u, v)
        else:
            raise ValueError("`nodes` must be either an int or a list of layer sizes.")

        nx.set_node_attributes(G, layers, 'layer')
        return G

    def generate_ws_graph(self, nodes, k=2, p=0.05):
        return nx.watts_strogatz_graph(nodes, k, p)
    
    def generate_ws_dag(self, nodes, first_layer_size=10, k=2, p=0.7):
        # Modified and slightly optimized implementation of the method mentioned in Structural Analysis of Sparse Neural Networks by Julian Stier and Michael Granitzer
        # Lower nodes are preferred since the graph can get deeper with higher node counts, which degrades performance
        # Try to generate a WS graph until it is connected
        while True:
            ws = nx.watts_strogatz_graph(nodes, k, p)
            if nx.is_connected(ws):
                break
        #DAG
        dag = nx.DiGraph()
        dag.add_nodes_from(range(nodes))
        for u, v in ws.edges():
            if u > v: #lower triangular part of the graph
                dag.add_edge(u, v)
        
        
        # The next phase is to build layer structure, firstly using shortest path distance to connect layers, then fallbacks to random assignment
        # Topological sort the graph to get a starting layer structure
        layers = {}
        for node in nx.topological_sort(dag):
            preds = list(dag.predecessors(node))
            if preds:
                layers[node] = max(layers[p] for p in preds)+1
            else:
                layers[node] = 0
        
        # Assign remaining layers based on shortest path distance from first layer
        remaining_nodes = set(range(first_layer_size, nodes))
        current_layer = 1
        
        while remaining_nodes:
            # Find nodes that connect to the previous layer
            next_layer_nodes = set()
            current_layer_candidates = set()
            
            for node in remaining_nodes:
                # check if this node has edges from previous layer
                has_prev_connection = any(
                    dag.has_edge(prev, node) 
                    for prev in dag.nodes() 
                    if prev in layers and layers[prev] == current_layer-1
                )
                
                if has_prev_connection:
                    current_layer_candidates.add(node)
            
            # If no nodes connect directly, use shortest path distance
            if not current_layer_candidates:
                for node in remaining_nodes:
                    min_dist = float('inf')
                    for src in dag.nodes():
                        if src in layers and layers[src] == current_layer-1:
                            try:
                                dist = nx.shortest_path_length(dag, src, node)
                                min_dist = min(min_dist, dist)
                            except nx.NetworkXNoPath:
                                continue
                    
                    if min_dist < float('inf'):
                        current_layer_candidates.add(node)
            # If no candidates are found, take a random sample of remaining nodes
            if not current_layer_candidates:
                # Take 10% of remaining nodes
                take_count = max(1, len(remaining_nodes) //10)
                current_layer_candidates = set(list(remaining_nodes)[:take_count])
            
            # Assign layer to the candidates from before
            for node in current_layer_candidates:
                layers[node] = current_layer
                next_layer_nodes.add(node)
            
            # Update
            remaining_nodes-= next_layer_nodes
            current_layer+= 1
            
            #safety check if we're not making progress. assign all remaining to final layer
            if not next_layer_nodes:
                for node in remaining_nodes:
                    layers[node] = current_layer
                break
        
        # This is here so that edges only go forward
        for u, v in list(dag.edges()):
            if layers[u] >= layers[v]:
                dag.remove_edge(u, v)

        nx.set_node_attributes(dag, layers, 'layer')
        return dag

    def generate_ba_dag(self, nodes, k, first_layer_size=10):


        ba_graph = nx.barabasi_albert_graph(nodes, k)
        #DAG
        dag = nx.DiGraph()
        dag.add_nodes_from(range(nodes))
        for u, v in ba_graph.edges():
            if u > v: #lower triangular part of the graph
                dag.add_edge(u, v)
        
        
        layers = {}
        for node in nx.topological_sort(dag):
            preds = list(dag.predecessors(node))
            if preds:
                layers[node] = max(layers[p] for p in preds)+1
            else:
                layers[node] = 0
        
        # Assign remaining layers based on shortest path distance from first layer
        remaining_nodes = set(range(first_layer_size, nodes))
        current_layer = 1
        
        while remaining_nodes:
            # Find nodes that connect to the previous layer
            next_layer_nodes = set()
            current_layer_candidates = set()
            
            for node in remaining_nodes:
                # check if this node has edges from previous layer
                has_prev_connection = any(
                    dag.has_edge(prev, node) 
                    for prev in dag.nodes() 
                    if prev in layers and layers[prev] == current_layer-1
                )
                
                if has_prev_connection:
                    current_layer_candidates.add(node)
            
            # If no nodes connect directly, use shortest path distance
            if not current_layer_candidates:
                for node in remaining_nodes:
                    min_dist = float('inf')
                    for src in dag.nodes():
                        if src in layers and layers[src] == current_layer-1:
                            try:
                                dist = nx.shortest_path_length(dag, src, node)
                                min_dist = min(min_dist, dist)
                            except nx.NetworkXNoPath:
                                continue
                    
                    if min_dist < float('inf'):
                        current_layer_candidates.add(node)
            
            if not current_layer_candidates:
                # Take 10% of remaining nodes
                take_count = max(1, len(remaining_nodes) // 10)
                current_layer_candidates = set(list(remaining_nodes)[:take_count])
            
            # Assign layer to the candidates from before
            for node in current_layer_candidates:
                layers[node] = current_layer
                next_layer_nodes.add(node)
            
            remaining_nodes-= next_layer_nodes
            current_layer+= 1
            
            if not next_layer_nodes:
                for node in remaining_nodes:
                    layers[node] = current_layer
                break
        
        # This is here so that edges only go forward
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
    

    def calculate_graph_metrics(self, G, model):
        if not G:
            return {}
        if model.named_parameters() is None:
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
        
        for name, param in model.named_parameters():
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
            metrics['betweenness'] = nx.betweenness_centrality(G)
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
