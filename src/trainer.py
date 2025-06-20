from src.training.neuralnetwork import MLPNet, LSTMNet, SparseMLPNet, SparseLSTMNet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from PyQt6.QtCore import QThread, pyqtSignal
from src.training.pruner import MagnitudePruner, RandomPruner, PrunerThread
from src.training.datahandler import DataHandler
from src.training.modelhandler import ModelHandler
import networkx as nx
from src.training.graphhandler import GraphHandler
import warnings
warnings.filterwarnings("ignore", message="The verbose parameter is deprecated.*")


class Trainer(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(self, model_type, dataset_type, lr=None, hidden_sizes=[6,6,6], loss="CrossEntropy", optimizer="Adam", epochs=30, k=2, p=0.05,graph_type="Full", N=250, batch_size=64, layer_count=5, index=None):

        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.model_type = model_type
        self.model = None
        self.lr = lr
        self.epochs = epochs
        self.optimizer_type = optimizer
        self.optimizer = None
        self.k = k
        self.p = p
        self.dataset_type = dataset_type
        self.graph_type = graph_type
        self.batch_size = batch_size
        self.layer_count = layer_count
        self.index = index
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if loss == "CrossEntropy":
            self.criterion = nn.CrossEntropyLoss()
        else: #for now default to MSELoss
            self.criterion = nn.MSELoss()
        self.running = True
        self.current_epoch = 0
        self.data_loaded = False
        self.training_metrics = {
            'final_train_loss': None,
            'final_train_accuracy': None,
            'final_val_loss': None,
            'final_val_accuracy': None,
            'model_metrics': {},
            'graph_metrics': {}
        }
        self.pruner_thread = None
        self.train_loader = None
        self.test_loader = None
        self.dag_graph = None
        self.download_thread = None

        
    def load_data_and_create_graph(self):
        data_handler = DataHandler(self.dataset_type, self.batch_size)

        download_thread = data_handler.load_data(message_callback=self.message.emit)
        download_thread.data_loaded.connect(self.handle_data_loaded)
        download_thread.error.connect(lambda msg: self.message.emit(f"Error: {msg}"))
        
        self.download_thread = download_thread

    def handle_data_loaded(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.data_loaded = True

        data_handler = DataHandler(self.dataset_type, self.batch_size)
        self.dataset_properties = data_handler.get_dataset_properties()
        
        input_size = self.dataset_properties["input_size"]
        num_classes = self.dataset_properties["num_classes"]
        self.feature_size = self.dataset_properties["feature_size"]
        self.sequence_length = self.dataset_properties["sequence_length"]

        self.graph_handler = GraphHandler()
        self.dag_graph = self.graph_handler.create_dag_graph(self.hidden_sizes, self.graph_type, self.k, self.p, self.layer_count)

        self.model_handler = ModelHandler(self.model_type, self.hidden_sizes, self.device)
        self.model = self.model_handler.create_model(self.graph_type, self.dag_graph, input_size, num_classes, self.feature_size)

        if self.graph_type == "WS" or self.graph_type == "BA":
            progress_message = f"Parameters of the {self.model_type} {self.graph_type} trainer: {self.epochs} epoch, {self.lr} learning_rate, {self.optimizer_type} optimizer, {self.criterion} loss function, {self.k} k, {self.p} p, {self.dataset_type} dataset, {self.hidden_sizes} hidden sizes"
        else:
            progress_message = f"Parameters of the {self.model_type} {self.graph_type} trainer: {self.epochs} epoch, {self.optimizer_type} optimizer, {self.criterion} loss function, {self.dataset_type} dataset, {self.hidden_sizes} hidden sizes"
        self.message.emit(progress_message)
        self.download_thread.finished.emit()



    def run(self):
        progress_message = "\nStarting the training process..."
        self.message.emit(progress_message)
        
        
        if not self.data_loaded:
            print("Error: Data not loaded")
            self.message.emit("Error: Data not loaded")
            return
            
        if self.model is None:
            print("Error: Model is None")
            self.message.emit("Error: Model is None")
            return
        
        if not self.data_loaded:
            print("Error: Data not loaded")
            self.message.emit("Error: Data not loaded")
            return
            
        if self.model is None:
            print("Error: Model is None")
            self.message.emit("Error: Model is None")
            return
            
        print(f"Starting training for model {self.index} with {self.epochs} epochs and {self.lr} learning rate")
        self.train(self.model, self.train_loader, self.epochs - self.current_epoch, lr=self.lr)
        self.finished.emit()  # notify gui when done
    
    def stop(self):
        self.running = False
        if self.pruner_thread and self.pruner_thread.isRunning():
            self.pruner_thread.terminate()
        self.terminate()
        self.wait()
        self.finished.emit()

    # Pruning methods
    def async_prune(self, prune_ratio, prune_type="FULL", prune_mode="Magnitude"):
        if self.pruner_thread and self.pruner_thread.isRunning():
            self.message.emit("Pruning already in progress")
            return False

        self.pruner_thread = PrunerThread(self.model, prune_ratio, prune_type, prune_mode)

        self.pruner_thread.progress_message.connect(self.handle_pruning_message)
        self.pruner_thread.validation_info.connect(self.handle_validation_info)
        self.pruner_thread.results_ready.connect(self.handle_pruning_results)
        self.pruner_thread.finished.connect(self.handle_pruning_finished)
        self.pruner_thread.start()
        return True
    
    def handle_pruning_message(self, message):
        print(message)
        self.message.emit(message)

    def on_pruning_complete(self, success):
        if success:
            self.message.emit("Pruning completed successfully")
        else:
            self.message.emit("Pruning failed")
        self.pruner_thread = None
   
    def handle_validation_info(self, info):
        print("\n=== Pre-Pruning Validation ===")
        print(f"Model type: {info['model_type']}")
        print(f"Device: {info['device']}")
        print(f"Parameter tensors: {info['parameter_tensors']}")
        
    def handle_pruning_results(self, results):
        print("\n=== Pruning Results ===")
        print(f"Actual sparsity: {results['actual_sparsity']:.2%} (target: {results['target_sparsity']:.2%})")
        
        self.training_metrics['model_metrics'] = {
            'total_parameters': results['total_parameters'],
            'global_sparsity_prune': results['actual_sparsity'],
            'prune_type': results['prune_type']
        }
        
    def handle_pruning_finished(self, success):
        self.pruner_thread = None
        
    # Training methods
    def train(self, model, train_loader, epochs=30, lr=0.001):
        print("Training started...")
        model.to(self.device)
        
        optimizer_configs = {
            "Adam": {"lr": 0.001 if lr is None else lr, "weight_decay": 1e-4},
            "SGD": {"lr": 0.01 if lr is None else lr, "momentum": 0.9, "weight_decay": 1e-4},
            "RMSprop": {"lr": 0.001 if lr is None else lr, "weight_decay": 1e-4},
            "Adadelta": {"lr": 1.0 if lr is None else lr, "weight_decay": 1e-4},
        }
        
        # Get optimizer config
        optimizer_config = optimizer_configs.get(self.optimizer_type, {"lr": 0.001, "weight_decay": 1e-4})
        if lr is not None:
            optimizer_config["lr"] = lr

        optimizers = {
            "Adam": optim.Adam,
            "SGD": optim.SGD,
            "RMSprop": optim.RMSprop,
            "Adadelta": optim.Adadelta
        }
        
        # optimizer
        self.optimizer = optimizers.get(self.optimizer_type, optim.Adam)(
            model.parameters(), 
            **optimizer_config
        )
        
        # Use learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=1, 
            verbose=True
        )

        for epoch in range(epochs):
            if not self.running:
                print("Training stopped early")
                return

            model.train()
            total_loss, correct, total = 0, 0, 0

            for images, labels in train_loader:
                if not self.running:
                    print("Training stopped early")
                    return
                images, labels = images.to(self.device), labels.to(self.device)

                #Reshape images for LSTM models
                if isinstance(model, (LSTMNet, SparseLSTMNet)):
                    if self.dataset_type == "MNIST":
                        images = images.view(images.size(0), self.sequence_length, self.feature_size)
                    elif self.dataset_type in ["CIFAR-10", "CIFAR-100"]:
                        images = images.view(images.size(0), self.sequence_length, -1)
                else:
                    images = images.view(images.size(0), -1)

                self.optimizer.zero_grad()
                outputs = model(images)
                
                if isinstance(self.criterion, nn.MSELoss):
                    labels_one_hot = torch.zeros(labels.size(0), self.dataset_properties["num_classes"]).to(self.device)
                    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
                    loss = self.criterion(outputs, labels_one_hot)
                else:
                    loss = self.criterion(outputs, labels)

                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Re-apply masks for sparse models after optimizer step
                if hasattr(model, '_apply_masks'):
                    model._apply_masks()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            self.current_epoch += 1
            train_loss = total_loss / len(train_loader)
            train_acc = 100. * correct / total
            val_loss, val_acc = self.validate(model)
            
            scheduler.step(val_loss)
            
            self.training_metrics.update({
                'final_train_loss': train_loss,
                'final_train_accuracy': train_acc,
                'final_val_loss': val_loss,
                'final_val_accuracy': val_acc
            })

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            self.message.emit(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")


        print("Training complete.")
        self.message.emit("Training complete.")

    def validate(self, model=None):
        model = model or self.model
        model.eval()
        

        total_loss, correct, total = 0, 0, 0
        
        if not hasattr(self, 'test_loader'):
            return float('nan'), 0.0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                

                if isinstance(model, (LSTMNet, SparseLSTMNet)):
                    if self.dataset_type == "MNIST":
                        images = images.view(images.size(0), self.sequence_length, self.feature_size)
                    elif self.dataset_type in ["CIFAR-10", "CIFAR-100"]:
                        images = images.view(images.size(0), self.sequence_length, -1)
                else:
                    images = images.view(images.size(0), -1)
                
                outputs = model(images)
                
                if isinstance(self.criterion, nn.MSELoss):
                    labels_one_hot = torch.zeros(labels.size(0), self.dataset_properties["num_classes"]).to(self.device)
                    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
                    loss = self.criterion(outputs, labels_one_hot)
                else:
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        if len(self.test_loader) == 0:
            return float('nan'), 0.0
        
        return total_loss / len(self.test_loader), 100. * correct / total

    # Results and state management methods
    def get_state(self):
        pruning_metrics = self.training_metrics['model_metrics']
        self.graph_metrics = self.graph_handler.calculate_graph_metrics(self.dag_graph, self.model) if hasattr(self, 'dag_graph') else {}
        model_metrics = self.model_handler.calculate_metrics() if hasattr(self, 'model') else {}
        self.training_metrics['model_metrics'].update(model_metrics)
        serialized_graph = self.model_handler.serialize_graph(self.dag_graph) if hasattr(self, 'dag_graph') else None
        return {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self.optimizer, 'state_dict') else None,
            'current_epoch': self.current_epoch,
            'index': self.index,
            'training_metrics': self.training_metrics,
            'hidden_sizes': self.hidden_sizes,
            'lr': self.lr,
            'model_type': self.model_type,
            'dataset_type': self.dataset_type,
            'graph_type': self.graph_type,
            'k': self.k,
            'p': self.p,
            'layer_count': self.layer_count,
            'batch_size': self.batch_size,
            'dag_graph': nx.node_link_data(self.dag_graph) if hasattr(self, 'dag_graph') else None,
            'model': self.model.__class__.__name__ if hasattr(self, 'model') else None,
            'optimizer_type': self.optimizer.__class__.__name__ if hasattr(self, 'optimizer') else None,
            'criterion': self.criterion.__class__.__name__ if hasattr(self, 'criterion') else None,
            'training_metrics': self.training_metrics if hasattr(self, 'training_metrics') else None,
            'graph_metrics': self.graph_metrics if hasattr(self, 'graph_metrics') else None,
            'prune_type': pruning_metrics.get('prune_type') if hasattr(self, 'pruning_metrics') else None,
            'input_size': self.dataset_properties.get('input_size') if hasattr(self, 'dataset_properties') else None,
            'num_classes': self.dataset_properties.get('num_classes') if hasattr(self, 'dataset_properties') else None,
            'feature_size': self.dataset_properties.get('feature_size') if hasattr(self, 'dataset_properties') else None,
            'sequence_length': self.dataset_properties.get('sequence_length') if hasattr(self, 'dataset_properties') else None,
            'serialized_graph': serialized_graph if hasattr(self, 'dag_graph') else None,
        }
    
    
    def set_state(self, state):
        self.current_epoch = state.get('current_epoch', 0)
        self.index = state.get('index')
        self.training_metrics = state.get('training_metrics', {})
        self.hidden_sizes = state.get('hidden_sizes', [])
        self.lr = state.get('lr')
        self.model_type = state.get('model_type')
        self.dataset_type = state.get('dataset_type')
        self.graph_type = state.get('graph_type')
        self.k = state.get('k', 2)
        self.p = state.get('p', 0.05)
        self.layer_count = state.get('layer_count', 5)
        self.batch_size = state.get('batch_size', 64)

        if self.model and state.get('model_state_dict'):
            self.model.load_state_dict(state['model_state_dict'])
            
        if self.model:
            optimizers = {
                "Adam": optim.Adam,
                "SGD": optim.SGD,
                "RMSprop": optim.RMSprop,
                "Adadelta": optim.Adadelta
            }
            opt_class = optimizers.get(self.optimizer_type, optim.Adam)
            self.optimizer = opt_class(self.model.parameters(), lr=self.lr or 0.001)
            if state.get('optimizer_state_dict'):
                self.optimizer.load_state_dict(state['optimizer_state_dict'])

        if state.get('dag_graph'):
            self.dag_graph = nx.node_link_graph(state['dag_graph'])
        if state.get('graph_metrics'):
            self.graph_metrics = state['graph_metrics']
        if state.get('model'):
            model_class = globals().get(state['model'])
            if model_class:
                self.model = model_class(self.hidden_sizes).to(self.device)
            else:
                raise ValueError(f"Model class {state['model']} not found")
        if state.get('training_metrics'):
            self.training_metrics = state['training_metrics']
        if state.get('graph_metrics'):
            self.graph_metrics = state['graph_metrics']
