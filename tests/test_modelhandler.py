import pytest
import networkx as nx
from unittest.mock import MagicMock
from src.training.modelhandler import ModelHandler


@pytest.fixture #this means that this function will be used as a fixed part in the test cases!
def mock_model_handler():
    return ModelHandler(model_type="MLP",hidden_sizes=[64,32],device="cpu")



def test_create_model_sparse_mlp(mock_model_handler):
    mock_dag = nx.DiGraph()
    nx.set_node_attributes(mock_dag, {0: {'layer':0}, 1: {'layer':1}})
    mock_model_handler.dag_to_mlp_structure = MagicMock(return_value=[2,3,1]) #magicmock isin general used to mock the behavior of an object in a test case
    mock_model_handler.model = MagicMock()
     
    
    mock_model_handler.create_model(graph_type="WS", dag_graph=mock_dag, input_size=2, output_size=1)
    assert mock_model_handler.model is not None

def test_create_model_sparse_lstm(mock_model_handler):
    mock_dag = nx.DiGraph()
    nx.set_node_attributes(mock_dag, {0: {'layer': 0}, 1: {'layer': 1}})
    mock_model_handler.dag_to_lstm_structure = MagicMock(return_value=[2, 3, 1])
    mock_model_handler.model = MagicMock()
    
    mock_model_handler.create_model(graph_type="WS", dag_graph=mock_dag, input_size=2, output_size=1, feature_size=2)
    assert mock_model_handler.model is not None

def test_calculate_metrics_no_model(mock_model_handler):
    metrics = mock_model_handler.calculate_metrics()
    assert metrics['total_parameters'] == 0
    assert metrics['trainable_parameters'] == 0
    assert metrics['global_sparsity'] == 0.0
    assert metrics['layer_sparsity'] == {}

def test_calculate_metrics_with_model(mock_model_handler):
    mock_model_handler.model = MagicMock()
    mock_model_handler.model.named_parameters = MagicMock(return_value=[
        ("layer1.weight", MagicMock(numel=MagicMock(return_value=100), requires_grad=True, __eq__=MagicMock(return_value=0))),
        ("layer2.bias", MagicMock(numel=MagicMock(return_value=50), requires_grad=False))
    ])
    metrics = mock_model_handler.calculate_metrics()
    assert metrics['total_parameters'] == 150
    assert metrics['trainable_parameters'] == 100
    assert metrics['global_sparsity'] == 0.0

def test_dag_to_mlp_structure(mock_model_handler):
    mock_dag = nx.DiGraph()
    mock_dag.add_node(0, layer=0)
    mock_dag.add_node(1, layer=1)
    mock_dag.add_node(2, layer=1)
    mock_dag.add_node(3, layer=2)
    structure = mock_model_handler.dag_to_mlp_structure(mock_dag, input_size=3, output_size=2)
    assert structure == [3, 2, 2]

    mock_dag = nx.DiGraph()
    structure = mock_model_handler.dag_to_mlp_structure(mock_dag, input_size=1, output_size=1)
    assert structure == [1, 1]

    mock_dag = nx.DiGraph()
    nx.set_node_attributes(mock_dag, {0: {'layer': 0}})
    structure = mock_model_handler.dag_to_mlp_structure(mock_dag, input_size=1, output_size=1)
    assert structure == [1, 1]

    mock_dag = nx.DiGraph()
    nx.set_node_attributes(mock_dag, {
        0: {'layer': 0},
        1: {'layer': 2}
    })
    structure = mock_model_handler.dag_to_mlp_structure(mock_dag, input_size=2, output_size=1)
    assert structure == [2,1]

    mock_dag = nx.DiGraph()
    mock_dag.add_node(0, layer=0)
    mock_dag.add_node(1, layer=1)
    mock_dag.add_node(2, layer=1)
    mock_dag.add_node(3, layer=2)
    mock_dag.add_node(4, layer=3)
    structure = mock_model_handler.dag_to_mlp_structure(mock_dag, input_size=4, output_size=2)
    assert structure == [4,2,1,2]

def test_dag_to_lstm_structure(mock_model_handler):
        mock_dag = nx.DiGraph()
        mock_dag.add_node(0, layer=0)
        mock_dag.add_node(1, layer=2)
        structure = mock_model_handler.dag_to_lstm_structure(mock_dag, input_size=2, output_size=1)
        assert structure == [2, 1, 0, 1,1]

        mock_dag = nx.DiGraph()
        mock_dag.add_node(0, layer=0)
        mock_dag.add_node(1, layer=1)
        mock_dag.add_node(2, layer=1)
        mock_dag.add_node(3, layer=2)
        structure = mock_model_handler.dag_to_lstm_structure(mock_dag, input_size=3, output_size=2)
        assert structure == [3, 1, 2, 1, 2]

        # no nodes
        mock_dag = nx.DiGraph()
        structure = mock_model_handler.dag_to_lstm_structure(mock_dag, input_size=1, output_size=1)
        assert structure == [1, 1]  # input and output

        mock_dag = nx.DiGraph()
        nx.set_node_attributes(mock_dag, {0: {'layer': 0}})
        structure = mock_model_handler.dag_to_lstm_structure(mock_dag)
        assert structure == []
