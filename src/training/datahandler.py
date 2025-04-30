import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PyQt6.QtCore import QThread, pyqtSignal, QObject
import os

class DatasetDownloadThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    data_loaded = pyqtSignal(object, object)
    
    def __init__(self, dataset_type, batch_size, root="./data"):
        super().__init__()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.root = root
        self.train_loader = None
        self.test_loader = None
        self.dataset_info = {
            "MNIST": {
                "dataset": datasets.MNIST,
                "transform": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            },
            "CIFAR-10": {
                "dataset": datasets.CIFAR10,
                "transform": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            },
            "CIFAR-100": {
                "dataset": datasets.CIFAR100,
                "transform": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            }
        }
        
    def run(self):
        try:
            if self.dataset_type not in self.dataset_info:
                raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
                
            dataset_config = self.dataset_info[self.dataset_type]
            transform = dataset_config["transform"]
            
            train_path = os.path.join(self.root, self.dataset_type.lower())
            if os.path.exists(train_path):
                self.progress.emit(f"Dataset {self.dataset_type} already exists. Loading...")
            else:
                self.progress.emit(f"Downloading {self.dataset_type} training set...")
                train_dataset = dataset_config["dataset"](
                    root=self.root, train=True, transform=transform, download=True
                )
                self.progress.emit(f"Downloading {self.dataset_type} test set...")
                test_dataset = dataset_config["dataset"](
                    root=self.root, train=False, transform=transform, download=True)
            
            self.progress.emit("Loading datasets...")
            train_dataset = dataset_config["dataset"](
                root=self.root, train=True, transform=transform, download=False
            )
            test_dataset = dataset_config["dataset"](
                root=self.root, train=False, transform=transform, download=False
            )

            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            self.progress.emit("Dataset loading complete!")
            self.data_loaded.emit(self.train_loader, self.test_loader)
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))

class DataHandler:
    def __init__(self, dataset_type, batch_size):
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.train_loader = None
        self.test_loader = None
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

    def load_data(self, message_callback=None):
        if self.dataset_type not in self.dataset_info:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        download_thread = DatasetDownloadThread(self.dataset_type, self.batch_size)
        
        if message_callback:
            download_thread.progress.connect(message_callback)
            download_thread.error.connect(lambda msg: message_callback(f"Error: {msg}"))
        
        download_thread.data_loaded.connect(self.set_data_loaders)
        
        return download_thread

    def set_data_loaders(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader

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