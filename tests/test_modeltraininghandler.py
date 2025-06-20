import pytest
from unittest.mock import MagicMock, patch
from PyQt6.QtGui import QMovie
from src.modeltraininghandler import ModelTrainingHandler
from src.trainer import Trainer
from PyQt6.QtTest import QSignalSpy

@pytest.fixture
def mock_window():
    return MagicMock()

@pytest.fixture
def handler(mock_window):
    return ModelTrainingHandler(mock_window)

def test_reset_UI(mock_window, handler):
    handler.reset_UI()
    
    mock_window.model_train_button.setEnabled.assert_called_with(True)
    mock_window.undo_button.setEnabled.assert_called_with(True)
    mock_window.model_train_button.setText.assert_called_with("Start Training")
    mock_window.model_train_button.clicked.connect.assert_called()



def test_parse_through_processes_table(mock_window, handler):
    mock_window.reorder_table_view2.model().get_table_data.return_value = [
        [1, "MLP", "Prior", "MNIST", "", "CrossEntropy", "Adam", 30, 2, 0.5, 64, 0.001, "WS"]
    ]

    handler.validate_table_data = MagicMock(return_value=True)
    handler.process_table_row = MagicMock(return_value=(True, None))
    handler.main_train_loop = MagicMock()

    handler.parse_through_processes_table()
    handler.process_table_row.assert_called_once_with(
        [1, "MLP", "Prior", "MNIST", "", "CrossEntropy", "Adam", 30, 2, 0.5, 64, 0.001, "WS"]
    )
    handler.main_train_loop.assert_called_once()



@patch('src.modeltraininghandler.Trainer')
def test_train_one_model(MockTrainer, mock_window, handler):
    mock_window.neural_networks = [[1, "MLP", "Prior", "MNIST", 250, "CrossEntropy", "Adam", 30, 2, 0.5, 64, 0.001, "WS"]]
    
    mock_trainer_instance = MagicMock(spec=Trainer)
    mock_trainer_instance.message = MagicMock()
    mock_trainer_instance.finished = MagicMock()
    
    mock_download_thread = MagicMock()
    mock_download_thread.finished = MagicMock()
    mock_trainer_instance.download_thread = mock_download_thread
    
    MockTrainer.return_value = mock_trainer_instance
    
    handler.train_one_model(mock_window.neural_networks[0])
    
    MockTrainer.assert_called_once()
    
    mock_trainer_instance.load_data_and_create_graph.assert_called_once()
    
    mock_trainer_instance.message.connect.assert_called_once()
    mock_download_thread.finished.connect.assert_called_once()
    callback = mock_download_thread.finished.connect.call_args[0][0]
    callback()
    mock_trainer_instance.start.assert_called_once()
    
    mock_trainer_instance.finished.connect.assert_called_once()


def test_handle_prune_action(mock_window, handler):
    row = [None, "Prune", "Layer1", 50, "Magnitude"]
    mock_trainer = MagicMock(spec=Trainer)
    mock_trainer.pruner_thread = MagicMock()
    mock_trainer.pruner_thread.finished = MagicMock()
    handler.trainer = mock_trainer
    handler.handle_prune_action(row)
    mock_trainer.async_prune.assert_called_with(0.5, "Layer1", "Magnitude")
    mock_trainer.pruner_thread.finished.connect.assert_called_once()
