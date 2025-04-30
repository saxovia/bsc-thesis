import pytest
from src.training.datahandler import DataHandler
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

def test_get_dataset_properties():
    handler_mnist = DataHandler(dataset_type="MNIST", batch_size=64)
    mnist_properties = handler_mnist.get_dataset_properties()
    assert mnist_properties == {
        "input_size": 28 * 28,
        "num_classes": 10,
        "feature_size": 28,
        "sequence_length": 28
    }

    handler_cifar10 = DataHandler(dataset_type="CIFAR-10", batch_size=32)
    cifar10_properties = handler_cifar10.get_dataset_properties()
    assert cifar10_properties == {
        "input_size": 3 * 32 * 32,
        "num_classes": 10,
        "feature_size": 32,
        "sequence_length": 32
    }

    handler_cifar100 = DataHandler(dataset_type="CIFAR-100", batch_size=128)
    cifar100_properties = handler_cifar100.get_dataset_properties()
    assert cifar100_properties == {
        "input_size": 3 * 32 * 32,
        "num_classes": 100,
        "feature_size": 32,
        "sequence_length": 32
    }

    handler_invalid = DataHandler(dataset_type="INVALID", batch_size=64)
    with pytest.raises(ValueError, match="Unsupported dataset type: INVALID"):
        handler_invalid.get_dataset_properties()

def test_load_data(qtbot):
    handler = DataHandler(dataset_type="MNIST", batch_size=64)
    download_thread = handler.load_data()
    
    train_loader = None
    test_loader = None
    
    def on_data_loaded(loaded_train, loaded_test):
        nonlocal train_loader, test_loader
        train_loader = loaded_train
        test_loader = loaded_test
    
    download_thread.data_loaded.connect(on_data_loaded)
    
    # Use qtbot to wait for the signal
    with qtbot.waitSignal(download_thread.data_loaded, timeout=10000):
        download_thread.start()
    
    assert train_loader is not None, "Train loader was not set"
    assert test_loader is not None, "Test loader was not set"
    
    assert len(train_loader) > 0
    assert len(test_loader) > 0

    for images, labels in train_loader:
        assert images.shape[0] in [32, 64]
        assert images.shape[1:] == (1, 28, 28)
        assert labels.shape[0] == images.shape[0]

    for images, labels in test_loader:
        assert images.shape[0] in [32, 64]
        assert images.shape[1:] == (1, 28, 28)
        assert labels.shape[0] == images.shape[0]
        break

def test_invalid_batch_sizes():
    # Test zero batch size
    with pytest.raises(ValueError, match="Batch size must be positive"):
        DataHandler(dataset_type="MNIST", batch_size=0)
    
    # Test negative batch size
    with pytest.raises(ValueError, match="Batch size must be positive"):
        DataHandler(dataset_type="MNIST", batch_size=-1)

def test_dataset_type_edge_cases():
    # Test empty string
    with pytest.raises(ValueError, match="Unsupported dataset type: "):
        DataHandler(dataset_type="", batch_size=64)
    
    # Test None
    with pytest.raises(ValueError, match="Unsupported dataset type: None"):
        DataHandler(dataset_type=None, batch_size=64)
    
    # Test case sensitivity
    with pytest.raises(ValueError, match="Unsupported dataset type: mnist"):
        DataHandler(dataset_type="mnist", batch_size=64)

def test_signal_handling(qtbot):
    handler = DataHandler(dataset_type="MNIST", batch_size=64)
    download_thread = handler.load_data()
    
    # Test multiple signal connections
    progress_messages = []
    def on_progress(msg):
        progress_messages.append(msg)
    
    download_thread.progress.connect(on_progress)
    download_thread.progress.connect(lambda msg: progress_messages.append(f"Second: {msg}"))
    
    # Test error signal
    error_message = None
    def on_error(msg):
        nonlocal error_message
        error_message = msg
    
    download_thread.error.connect(on_error)
    
    # Use qtbot to wait for the signal
    with qtbot.waitSignal(download_thread.data_loaded, timeout=10000):
        download_thread.start()
    
    assert len(progress_messages) > 0, "Progress signals were not emitted"
    assert error_message is None, "Error signal was unexpectedly emitted"

def test_concurrent_loading(qtbot):
    # Test loading the same dataset from multiple handlers
    handler1 = DataHandler(dataset_type="MNIST", batch_size=64)
    handler2 = DataHandler(dataset_type="MNIST", batch_size=64)
    
    download_thread1 = handler1.load_data()
    download_thread2 = handler2.load_data()
    
    train_loader1 = None
    test_loader1 = None
    train_loader2 = None
    test_loader2 = None
    
    def on_data_loaded1(loaded_train, loaded_test):
        nonlocal train_loader1, test_loader1
        train_loader1 = loaded_train
        test_loader1 = loaded_test
    
    def on_data_loaded2(loaded_train, loaded_test):
        nonlocal train_loader2, test_loader2
        train_loader2 = loaded_train
        test_loader2 = loaded_test
    
    download_thread1.data_loaded.connect(on_data_loaded1)
    download_thread2.data_loaded.connect(on_data_loaded2)
    
    # Use qtbot to wait for both signals
    with qtbot.waitSignal(download_thread1.data_loaded, timeout=10000), \
         qtbot.waitSignal(download_thread2.data_loaded, timeout=10000):
        download_thread1.start()
        download_thread2.start()
    
    assert train_loader1 is not None and test_loader1 is not None, "First loader failed"
    assert train_loader2 is not None and test_loader2 is not None, "Second loader failed"

def test_data_validation(qtbot):
    handler = DataHandler(dataset_type="MNIST", batch_size=64)
    download_thread = handler.load_data()
    
    train_loader = None
    test_loader = None
    
    def on_data_loaded(loaded_train, loaded_test):
        nonlocal train_loader, test_loader
        train_loader = loaded_train
        test_loader = loaded_test
    
    download_thread.data_loaded.connect(on_data_loaded)
    
    with qtbot.waitSignal(download_thread.data_loaded, timeout=10000):
        download_thread.start()
    
    for images, labels in train_loader:
        assert isinstance(images, torch.Tensor), "Images should be a tensor"
        assert isinstance(labels, torch.Tensor), "Labels should be a tensor"
        assert images.dtype == torch.float32, "Images should be float32"
        assert labels.dtype == torch.int64, "Labels should be int64"
        assert images.shape[1:] == (1, 28, 28), "Image shape should be (1, 28, 28)"
        assert labels.shape[0] == images.shape[0], "Number of labels should match number of images"
        break # First batch is enoguh
