
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataHandler:
    def __init__(self, dataset_type, batch_size):
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.dataset_info = {
            "MNIST": {
                "dataset": datasets.MNIST,
                "input_size": 28 * 28,
                "num_classes": 10,
                "transform": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]),
                "default_batch_size": 64,
                "feature_size" : 28,
                "sequence_length" : 28
            },
            "CIFAR-10": {
                "dataset": datasets.CIFAR10,
                "input_size": 3 * 32 * 32,
                "num_classes": 10,
                "transform": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]),
                "default_batch_size": 32,
                "feature_size" : 32,
                "sequence_length" : 32
            },
            "CIFAR-100": {
                "dataset": datasets.CIFAR100,
                "input_size": 3 * 32 * 32,
                "num_classes": 100,
                "transform": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]),
                "default_batch_size": 128,
                "feature_size" : 32,
                "sequence_length" : 32
            }
        }

    def load_data(self):
        if self.dataset_type not in self.dataset_info:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        dataset_config = self.dataset_info[self.dataset_type]
        transform = dataset_config["transform"]

        train_dataset = dataset_config["dataset"](
            root="./data", train=True, transform=transform, download=True
        )
        test_dataset = dataset_config["dataset"](
            root="./data", train=False, transform=transform, download=True
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def get_dataset_properties(self):
        if self.dataset_type not in self.dataset_info:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        dataset_config = self.dataset_info[self.dataset_type]
        return {
            "input_size": dataset_config["input_size"],
            "num_classes": dataset_config["num_classes"],
            "feature_size": dataset_config["feature_size"],
            "sequence_length": dataset_config["sequence_length"]
        }