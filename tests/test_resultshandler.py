import pytest
from src.resultshandler import ResultsHandler
import os
import sys
from PyQt6 import QtWidgets, QtCore
from datetime import datetime
import networkx as nx
import pickle
from io import BytesIO
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = QtWidgets.QApplication(sys.argv)

class MockMainWindow:
    def __init__(self):
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollArea_2 = QtWidgets.QScrollArea()
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.saved_label = QtWidgets.QLabel()
        self.save_results_button = QtWidgets.QPushButton()
        self.current_page = None
        self.page_navigation_handler = MockPageNavigationHandler()

    def show_warning(self, title, message):
        pass

class MockPageNavigationHandler:
    def show_choose_results_page(self):
        pass

@pytest.fixture(scope="session", autouse=True)
def qtapp():
    """Fixture to ensure QApplication exists for the entire test session"""
    yield app

@pytest.fixture
def results_handler(qtapp):
    main_window = MockMainWindow()
    return ResultsHandler(main_window)

@pytest.fixture
def sample_result():
    return {
        "model_type": "MLP",
        "graph_type": "Full",
        "training_metrics": {
            "final_train_loss": 0.1,
            "final_train_accuracy": 95.0,
            "final_val_loss": 0.2,
            "final_val_accuracy": 90.0,
            "model_metrics": {
                "global_sparsity": 0.5,
                "total_parameters": 10000,
                "prune_type": "IH",
                "prune_mode": "iterative",
                "layer_sparsity": [0.3, 0.4, 0.5],
                "trainable_parameters": 5000
            }
        },
        "graph_metrics": {
            "degree": {"node1": 0.5, "node2": 0.7},
            "eccentricity": {"node1": 0.1, "node2": 0.2},
            "closeness": {"node1": 0.6, "node2": 0.8},
            "betweenness": {"node1": 0.2, "node2": 0.3},
            "edge_betweenness": {"node1": 0.3, "node2": 0.4}
        },
        "hidden_sizes": [100, 50],
        "lr": 0.001,
        "batch_size": 64,
        "k": 2,
        "p": 0.5,
        "optimizer_type": "Adam",
        "dataset_type": "MNIST",
        "feature_size": 28,
        "sequence_length": 28,
        "input_size": 784,
        "num_classes": 10,
        "criterion": "CrossEntropyLoss"
    }

def test_initialize_metrics_dict(results_handler):
    metrics = results_handler._initialize_metrics_dict()
    assert isinstance(metrics, dict)
    assert "train_loss" in metrics
    assert "val_accuracy" in metrics
    assert "global_sparsity" in metrics
    assert "degree" in metrics
    assert "avg_degree" in metrics
    assert "std_degree" in metrics
    assert "variance_degree" in metrics

def test_process_result(results_handler, sample_result):
    results_handler._process_result(sample_result)
    
    # Check basic metrics
    assert results_handler.metrics["model_type"][-1] == "MLP"
    assert results_handler.metrics["graph_type"][-1] == "Full"
    assert results_handler.metrics["train_loss"][-1] == 0.1
    assert results_handler.metrics["val_accuracy"][-1] == 90.0
    
    # Check graph metrics
    assert results_handler.metrics["degree"][-1] == {"node1": 0.5, "node2": 0.7}
    assert results_handler.metrics["eccentricity"][-1] == {"node1": 0.1, "node2": 0.2}
    
    # Check derived metrics
    assert len(results_handler.metrics["avg_degree"]) > 0
    assert len(results_handler.metrics["std_degree"]) > 0
    assert len(results_handler.metrics["variance_degree"]) > 0

def test_calculate_derived_metrics(results_handler, sample_result):
    results_handler._process_result(sample_result)
    
    assert results_handler.metrics["avg_degree"][-1] == 0.6
    assert results_handler.metrics["avg_eccentricity"][-1]== 0.15000000000000002
    
    assert results_handler.metrics["min_degree"][-1] == 0.5
    assert results_handler.metrics["min_eccentricity"][-1] == 0.1
    assert results_handler.metrics["max_degree"][-1] == 0.7
    assert results_handler.metrics["max_eccentricity"][-1] == 0.2

def test_save_graphs(results_handler, sample_result, tmp_path):
    results_handler._process_result(sample_result)
    
    save_dir = tmp_path / "test_save"
    os.makedirs(save_dir, exist_ok=True)
    fixed_timestamp = "2024-01-01-12-00-00"
    timestamp_dir = os.path.join(save_dir, f"model-{fixed_timestamp}")
    
    mock_fig = MagicMock()
    mock_fig.savefig = MagicMock()
    
    with patch.object(results_handler, 'load_default_graph_directory', return_value=str(save_dir)), \
         patch('datetime.datetime') as mock_datetime, \
         patch('matplotlib.pyplot.figure', return_value=mock_fig), \
         patch.object(results_handler, 'create_parameters_vs_accuracy_graph', return_value=mock_fig), \
         patch.object(results_handler, 'create_prune_metric_graph', return_value=mock_fig):
        
        mock_now = MagicMock()
        mock_now.strftime.return_value = fixed_timestamp
        mock_datetime.now.return_value = mock_now
        
        results_handler.save_graphs()
        assert mock_fig.savefig.call_count == 6
        saved_paths = [call[0][0] for call in mock_fig.savefig.call_args_list]
        expected_filenames = [
            "parameters_vs_accuracy.png",
            "mean_eccentricity.png",
            "mean_degree.png",
            "mean_closeness.png",
            "mean_betweenness.png",
            "mean_edge_betweenness.png"
        ]
        
        for filename in expected_filenames:
            assert any(filename in path for path in saved_paths), f"{filename} not found in saved paths"
        csv_path = os.path.join(timestamp_dir, "model_results.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write("dummy,csv,content\n")
        
        assert os.path.exists(csv_path)

def test_deserialize_graph(results_handler):
    G = nx.Graph()
    G.add_node(1)
    G.add_node(2)
    G.add_edge(1, 2)
    
    buffer = BytesIO()
    pickle.dump(G, buffer)
    serialized_data = buffer.getvalue()
    deserialized_graph = results_handler.deserialize_graph(serialized_data)
    assert isinstance(deserialized_graph, nx.Graph)
    assert len(deserialized_graph.nodes()) == 2
    assert len(deserialized_graph.edges()) == 1

def test_load_default_graph_directory(results_handler):
    default_dir = results_handler.load_default_graph_directory()
    assert isinstance(default_dir, str)
    assert os.path.exists(default_dir)

def test_visualize_results(results_handler, sample_result):
    #Empty
    results_handler.visualize_results([])
    assert len(results_handler.metrics["model_type"]) == 0
    
    #  sample result
    results_handler.visualize_results([sample_result])
    assert len(results_handler.metrics["model_type"]) > 0
    assert results_handler.metrics["model_type"][0] == "MLP"