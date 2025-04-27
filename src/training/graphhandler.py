from collections import defaultdict
import networkx as nx

class GraphHandler:
    def __init__(self):
        pass

    def create_dag_graph(self, nodes, graph_type, k=2, p=0.05, target_layers=5, layer_count=5):
        if graph_type == "Full":
            return self.generate_fully_connected_graph(nodes)
        elif graph_type == "WS":
            return self.generate_ws_dag(nodes, k=k, p=p, target_layers=target_layers)
        elif graph_type == "BA":
            return self.generate_ba_dag(nodes, edges_per_node=k, target_layers=target_layers)
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")

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
    

    def generate_ba_dag(self, nodes, k, target_layers=5):
        ba_graph = nx.barabasi_albert_graph(nodes, k)
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
    