import networkx as nx

class NetworkPrior:
    """Basic abstract class for Network Priors."""
    def generate(self, *args, **kwargs):
        """Method to generate the network prior structure."""
        raise NotImplementedError("The network prior must be implemented in subclasses")
    
    def convert_to_dag(G):
        """Basic method for converting an undirected graph to a DAG."""
        adj_matrix = nx.to_numpy_array(G)
        n = adj_matrix.shape[0]
        dag = nx.DiGraph()
        dag.add_nodes_from(G.nodes)
        for i in range(n):
            for j in range(i):
                if adj_matrix[i][j] > 0:
                    dag.add_edge(i, j)
        return dag

class WattsStrogatzPrior(NetworkPrior):
    """Watts-Strogatz Prior."""
    def generate(self, N=250, k=2, p=0.8):
        import networkx as nx
        ws_graph = nx.watts_strogatz_graph(N, k, p) 
        return self.convert_to_dag(ws_graph)

class BarabasiAlbertPrior(NetworkPrior):
    """Barabasi-Albert Prior."""
    def generate(self, N=250, m=2):
        import networkx as nx
        ba_graph = nx.barabasi_albert_graph(N, m) 
        return self.convert_to_dag(ba_graph)
