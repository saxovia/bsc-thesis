import pytest
from PyQt6.QtCore import QThread
from src.trainer import Trainer
from unittest.mock import Mock, patch, MagicMock
import os, sys
from PyQt6.QtTest import QSignalSpy
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def trainer():
    return Trainer(
        model_type="MLPNet",
        dataset_type="MNIST",
        lr=0.001,
        hidden_sizes=[64, 64],
        loss="CrossEntropy",
        optimizer="Adam",
        epochs=10,
        k=2,
        p=0.1,
        graph_type="Full",
        batch_size=32,
        layer_count=3
    )

def test_trainer_initialization(trainer):
    assert trainer.model_type == "MLPNet"
    assert trainer.dataset_type == "MNIST"
    assert trainer.lr == 0.001
    assert trainer.hidden_sizes == [64, 64]
    assert trainer.epochs == 10
    assert trainer.optimizer_type == "Adam"
    assert trainer.graph_type == "Full"
    assert trainer.batch_size == 32
    assert trainer.layer_count == 3

def test_load_data_and_create_graph(trainer, mocker):
    mock_data_handler_instance = mocker.Mock()
    mock_data_handler_instance.load_data.return_value = ("mock_train_loader", "mock_test_loader")
    mock_data_handler_instance.dataset_info = {
        "MNIST": {
            "input_size": 28*28,
            "num_classes": 10,
            "feature_size": 28,
            "sequence_length": 28,
        }
    }
    mock_data_handler = mocker.patch("src.trainer.DataHandler", return_value=mock_data_handler_instance)
    mock_graph_handler_instance = mocker.Mock()
    mock_graph_handler_instance.create_dag_graph.return_value = "mock_dag_graph"
    mock_graph_handler = mocker.patch("src.trainer.GraphHandler", return_value=mock_graph_handler_instance)

    mock_model_handler_instance = mocker.Mock()
    mock_model_handler_instance.create_model.return_value = "mock_model"
    mock_model_handler = mocker.patch("src.trainer.ModelHandler", return_value=mock_model_handler_instance)

    trainer.load_data_and_create_graph()

    mock_data_handler.assert_called_once_with("MNIST", 32)
    mock_graph_handler.return_value.create_dag_graph.assert_called_once()
    mock_model_handler.return_value.create_model.assert_called_once()

def test_run_training(trainer, mocker):
    mock_train = mocker.patch.object(trainer, "train")


    trainer.message = mocker.Mock()
    trainer.finished = mocker.Mock()
    trainer.run()
    trainer.message.emit.assert_called_with("Starting the training process...")
    mock_train.assert_called_once()
    trainer.finished.emit.assert_called_once()


def test_stop_training(trainer, mocker):
    mock_terminate = mocker.patch.object(trainer, "terminate")
    mock_wait = mocker.patch.object(trainer, "wait")

    trainer.finished = mocker.Mock()
    trainer.stop()

    assert not trainer.running
    mock_terminate.assert_called_once()
    mock_wait.assert_called_once()
    trainer.finished.emit.assert_called_once()

def test_magnitude_prune(trainer, mocker):
    mock_async_prune = mocker.patch.object(trainer, "async_prune")

    trainer.magnitude_prune(0.2, mode="FULL")

    mock_async_prune.assert_called_once_with(0.2, 'FULL')


def test_async_prune(trainer, mocker):
    trainer.pruning=False

    mock_pruner_thread = mocker.patch("src.trainer.PrunerThread")
    spy=QSignalSpy(trainer.message)
    trainer.async_prune(0.2, mode="FULL")
    mock_pruner_thread.assert_called_once()
    assert len(spy)==0


def test_train(trainer, mocker):
    mock_model = Mock()
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
    mock_model.return_value = torch.randn(2, 10, requires_grad=True)  # logits

    mock_train_loader = [(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))]
    mock_validate = mocker.patch.object(trainer, "validate")
    mock_validate.return_value = (0.5, 80.0)

    spy = QSignalSpy(trainer.message)

    trainer.train(mock_model, mock_train_loader, epochs=1, lr=0.001)




def test_validate(trainer, mocker):
    mock_model = MagicMock()
    trainer.criterion = torch.nn.CrossEntropyLoss()
    trainer.device = torch.device("cpu")
    mock_model.return_value = torch.randn(2, 10, requires_grad=True)
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([1, 0])
    mock_test_loader = MagicMock()
    mock_test_loader.__iter__.return_value = iter([(images, labels)])
    trainer.test_loader = mock_test_loader

    val_loss, val_acc = trainer.validate(mock_model)

    assert isinstance(val_loss, float)
    assert isinstance(val_acc, float)



"""
def test_get_state(trainer):
    state = trainer.get_state()

    assert isinstance(state, dict)
    assert "model_state_dict" in state
    assert "optimizer_state_dict" in state
    assert "current_epoch" in state

def test_set_state(trainer, mocker):
    mock_model = Mock()
    trainer.model = mock_model

    state = {
        "current_epoch": 5,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "hidden_sizes": [64, 64],
        "lr": 0.001,
        "model_type": "MLPNet",
        "dataset_type": "MNIST",
        "graph_type": "Full",
        "k": 2,
        "p": 0.1,
        "layer_count": 3,
        "batch_size": 32
    }

    trainer.set_state(state)

    assert trainer.current_epoch == 5
    assert trainer.hidden_sizes == [64, 64]
    assert trainer.lr == 0.001
    assert trainer.model_type == "MLPNet
"""