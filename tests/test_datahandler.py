import pytest
from src.training.datahandler import DataHandler


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

def test_load_data():
    handler = DataHandler(dataset_type="MNIST", batch_size=64)
    train_loader, test_loader = handler.load_data()

    assert len(train_loader) > 0
    assert len(test_loader) > 0

    for images, labels in train_loader:
        assert images.shape == (64, 1, 28, 28)
        assert labels.shape == (64,)

    for images, labels in test_loader:
        assert images.shape == (64, 1, 28, 28)
        assert labels.shape == (64,)
        break

def get_dataset_properties():
    handler = DataHandler(dataset_type="MNIST", batch_size=64)
    properties = handler.get_dataset_properties()
    assert properties["input_size"] == 28 * 28
    assert properties["num_classes"] == 10
    assert properties["feature_size"] == 28
    assert properties["sequence_length"] == 28