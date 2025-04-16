import pytest
import networkx as nx
from src.training.graphhandler import GraphHandler
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class MockModel:

    def __init__(self, parameters):
        self.parameters = parameters

    def named_parameters(self):
        for name, param in self.parameters.items():
            yield name, param
class MockTensor:

    def __init__(self, data, requires_grad=True):
        self.data = data
        self.requires_grad = requires_grad

    def numel(self):
        return len(self.data)

    def __eq__(self, other):
        return MockTensor([1 if x == other else 0 for x in self.data])

    def sum(self):
        return MockTensor([sum(self.data)])

    def item(self):
        return sum(self.data)



def test_calculate_graph_metrics():
    handler = GraphHandler()
    #mock directed graph
    G = nx.DiGraph()
    G.add_edges_from([(0,1),(1,2),(2,3)])
    nx.set_node_attributes(G, {0: 0, 1: 1, 2: 2, 3: 3}, 'layer')

    model = MockModel({
        "layer1.weight": MockTensor([0,1,0,1]),
        "layer2.weight": MockTensor([1,1,0,0]),
        "bias": MockTensor([1,1], requires_grad=False)
    })
    metrics = handler.calculate_graph_metrics(G, model)

    assert metrics["total_parameters"] == 10
    assert metrics["trainable_parameters"] == 8
    assert metrics["global_sparsity"] == 0.5
    assert metrics["layer_sparsity"] == {
        "layer1.weight":0.5,
        "layer2.weight":0.5
    }
    assert "edge_betweenness" in metrics
    assert "closeness" in metrics
    assert "degree" in metrics

def test_calculate_graph_metrics_empty_graph():
    
    handler = GraphHandler()
    G = nx.DiGraph()
    model = MockModel({})

    metrics = handler.calculate_graph_metrics(G, model)
    assert metrics == {}

def test_calculate_graph_metrics_no_parameters():
    handler= GraphHandler()
    G = nx.DiGraph()
    G.add_edges_from([(0,1),(1,2)])
    model = MockModel({})

    metrics=handler.calculate_graph_metrics(G, model)

    assert metrics["total_parameters"]==0
    assert metrics["trainable_parameters"]==0
    assert metrics["global_sparsity"]==0.0