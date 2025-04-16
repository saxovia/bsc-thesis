import pytest
from unittest.mock import MagicMock
from src.training.pruner import PrunerThread, MagnitudePruner
import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
@pytest.fixture
def mock_model():
    mock_param = MagicMock()
    mock_param.numel = MagicMock(return_value=100)
    mock_param.__eq__ = MagicMock(return_value=0)
    mock_param.sum = MagicMock(return_value=0)
    mock_param.device = "cpu"
    model = MagicMock()
    model.parameters = MagicMock(return_value=iter([mock_param] * 3))
    return model

def test_run_model_attribute_missing():
    pruner_thread = PrunerThread(None, prune_ratio=0.5, mode="FULL")
    del pruner_thread.model #?
    pruner_thread.progress_message = MagicMock()
    pruner_thread.finished = MagicMock()

    pruner_thread.run()

    pruner_thread.progress_message.emit.assert_called_with("Pruning failed: Model attribute missing")
    pruner_thread.finished.emit.assert_called_with(False)

def test_run_model_is_string():
    pruner_thread = PrunerThread("dummy_model", prune_ratio=0.5, mode="FULL")
    pruner_thread.progress_message = MagicMock()
    pruner_thread.finished = MagicMock()

    pruner_thread.run()

    pruner_thread.progress_message.emit.assert_called_with("Pruning failed: Model is still a string ('dummy_model')")
    pruner_thread.finished.emit.assert_called_with(False)

def test_run_model_is_none():
    pruner_thread = PrunerThread(None, prune_ratio=0.5, mode="FULL")
    pruner_thread.progress_message = MagicMock()
    pruner_thread.finished = MagicMock()

    pruner_thread.run()
    pruner_thread.progress_message.emit.assert_called_with("Pruning failed: Model is None")
    pruner_thread.finished.emit.assert_called_with(False)

def test_run_model_has_no_parameters():
    mock_model = MagicMock()
    mock_model.parameters = MagicMock(return_value=iter([]))
    pruner_thread = PrunerThread(mock_model, prune_ratio=0.5, mode="FULL")
    pruner_thread.progress_message = MagicMock()
    pruner_thread.finished = MagicMock()

    pruner_thread.run()

    pruner_thread.progress_message.emit.assert_called_with("Pruning failed: Model exists but has no parameters")
    pruner_thread.finished.emit.assert_called_with(False)

def test_run_successful_pruning(mock_model):
    pruner_thread = PrunerThread(mock_model, prune_ratio=0.5, mode="FULL")
    pruner_thread.progress_message = MagicMock()
    pruner_thread.validation_info = MagicMock()
    pruner_thread.results_ready = MagicMock()
    pruner_thread.finished = MagicMock()

    pruner_thread.run()

    pruner_thread.progress_message.emit.assert_called_with("\nApplying FULL pruning at 50% ratio")
    pruner_thread.validation_info.emit.assert_called()
    pruner_thread.results_ready.emit.assert_called()
    pruner_thread.finished.emit.assert_called_with(True)
    



    pruner_thread.finished.emit.assert_called_with(False)

def test_run_model_is_none():
    pruner_thread = PrunerThread(None, prune_ratio=0.5, mode="FULL")
    pruner_thread.progress_message = MagicMock()
    pruner_thread.finished = MagicMock()

    pruner_thread.run()

    pruner_thread.progress_message.emit.assert_called_with("Pruning failed: Model is None")
    pruner_thread.finished.emit.assert_called_with(False)

def test_run_model_has_no_parameters(mock_model):
    mock_model.parameters = MagicMock(return_value=iter([]))
    pruner_thread = PrunerThread(mock_model, prune_ratio=0.5, mode="FULL")
    pruner_thread.progress_message = MagicMock()
    pruner_thread.finished = MagicMock()

    pruner_thread.run()

    pruner_thread.progress_message.emit.assert_called_with('Pruning failed: Parameter access failed: Model exists but has no parameters')
    pruner_thread.finished.emit.assert_called_with(False)

def test_run_successful_pruning(mock_model):
    mock_param1 = MagicMock()
    mock_param1.device = torch.device('cpu')
    mock_param1.__eq__.return_value = MagicMock(sum=MagicMock(return_value=MagicMock(item=MagicMock(return_value=10))))
    mock_param1.numel.return_value = 100

    mock_param2 = MagicMock()
    mock_param2.device = torch.device('cpu')
    mock_param2.__eq__.return_value = MagicMock(sum=MagicMock(return_value=MagicMock(item=MagicMock(return_value=20))))
    mock_param2.numel.return_value = 100
    mock_param3 = MagicMock()
    mock_param3.device = torch.device('cpu')
    mock_param3.__eq__.return_value = MagicMock(sum=MagicMock(return_value=MagicMock(item=MagicMock(return_value=30))))
    mock_param3.numel.return_value = 100
    mock_model.parameters.side_effect = lambda: iter([mock_param1, mock_param2, mock_param3])
    mock_model.named_parameters.return_value = [("weight1", mock_param1),("weight2", mock_param2),("weight3", mock_param3)]

    mock_model.__class__.__name__ = "MockModel"

    pruner_thread = PrunerThread(mock_model, prune_ratio=0.5, mode="FULL")
    pruner_thread.progress_message = MagicMock()
    pruner_thread.validation_info = MagicMock()
    pruner_thread.results_ready = MagicMock()
    pruner_thread.finished = MagicMock()

    pruner_thread.run()

    pruner_thread.validation_info.emit.assert_called_once_with({
        'model_type': 'MockModel',
        'device': 'cpu',
        'parameter_tensors':3
    })
    pruner_thread.results_ready.emit.assert_called_once()
    results = pruner_thread.results_ready.emit.call_args[0][0]
    assert results['total_parameters'] == 100 * 3
    assert results['prune_type'] == "FULL"
    assert results['target_sparsity'] == 0.5
    assert 0 <= results['actual_sparsity'] <= 1
 
    pruner_thread.finished.emit.assert_called_with(True)

def test_run_hh_pruning(mock_model):
    class MockTensor:
        def __eq__(self, other):
            result = MagicMock()
            result.sum.return_value = MagicMock()
            result.sum.return_value.item.return_value=25 if "hh" in self.name else 0
            return result
        
        def numel(self):
            return 100

    weight_hh = MockTensor()
    weight_hh.name = "weight_hh_l0"
    weight_hh.device = torch.device('cpu')
    
    weight_ih = MockTensor()
    weight_ih.name = "weight_ih_l0" 
    weight_ih.device = torch.device('cpu')

    mock_model.lstm = MagicMock(spec=torch.nn.LSTM)
    mock_model.parameters.side_effect = lambda: iter([weight_ih, weight_hh])
    mock_model.named_parameters.return_value = [
        ("lstm.weight_ih_l0", weight_ih),
        ("lstm.weight_hh_l0", weight_hh)
    ]
    mock_model.__class__.__name__ = "LSTMModel"
    pruner_thread = PrunerThread(mock_model, prune_ratio=0.4, mode="HH")
    pruner_thread.progress_message = MagicMock()
    pruner_thread.results_ready = MagicMock()
    pruner_thread.finished = MagicMock()
    
    pruner_thread.run()

    results = pruner_thread.results_ready.emit.call_args[0][0]
    assert results['prune_type']=="HH"
    assert results['actual_sparsity']==0.125
    pruner_thread.finished.emit.assert_called_with(True)


def test_run_ih_pruning(mock_model):
    class MockTensor:
        def __init__(self, name):
            self.name = name
            self.device = torch.device('cpu')
            
        def __eq__(self, other):
            result = MagicMock()
            result.sum.return_value = MagicMock(item=MagicMock(return_value=30 if "ih" in self.name else 0))
            return result
            
        def numel(self):
            return 100

    weight_ih = MockTensor("weight_ih_l0")
    weight_hh = MockTensor("weight_hh_l0")

    mock_model.lstm = MagicMock(spec=torch.nn.LSTM)
    mock_model.parameters.side_effect = lambda: iter([weight_ih, weight_hh])
    mock_model.named_parameters.return_value = [
        ("lstm.weight_ih_l0", weight_ih),
        ("lstm.weight_hh_l0", weight_hh)
    ]
    mock_model.__class__.__name__ = "LSTMModel"

    pruner_thread = PrunerThread(mock_model, prune_ratio=0.3, mode="IH")
    pruner_thread.progress_message = MagicMock()
    pruner_thread.results_ready = MagicMock()
    pruner_thread.run()

    results = pruner_thread.results_ready.emit.call_args[0][0]
    assert results['prune_type']=="IH"
    assert results['actual_sparsity']==0.15

def test_run_ho_pruning(mock_model):
    class MockTensor:
        def __init__(self, layer_type):
            self.layer_type = layer_type
            self.device = torch.device('cpu')
            
        def __eq__(self, other):
            result = MagicMock()
            result.sum.return_value = MagicMock(item=MagicMock(
                return_value=40 if self.layer_type == "output" else 0
            ))
            return result
            
        def numel(self):
            return 100

    hidden_layer = MockTensor("hidden")
    output_layer = MockTensor("output")

    mock_model.parameters.side_effect = lambda:iter([hidden_layer, output_layer])
    mock_model.named_parameters.return_value = [
        ("hidden.weight", hidden_layer),
        ("output.weight", output_layer)
    ]
    mock_model.__class__.__name__ = "MLPModel"

    pruner_thread = PrunerThread(mock_model, prune_ratio=0.5, mode="HO")
    pruner_thread.progress_message = MagicMock()
    pruner_thread.validation_info = MagicMock()
    pruner_thread.results_ready = MagicMock()
    pruner_thread.finished = MagicMock()
    pruner_thread.run()

    results = pruner_thread.results_ready.emit.call_args[0][0]
    assert results['prune_type']=="HO"
    assert results['actual_sparsity']==0.2
    assert results['target_sparsity']==0.5