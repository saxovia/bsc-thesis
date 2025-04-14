import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from src.neuralnetwork import MLPNet, LSTMNet
from PyQt6.QtCore import QThread, pyqtSignal

class BasePruner(ABC): #abstract class for pruning
    
    @abstractmethod
    def compute_mask(self, weight_tensor, prune_percent):
        pass
    
    def apply_pruning(self, model, prune_percent, mode="FULL"):
        if isinstance(model, nn.LSTM) or (hasattr(model, 'lstm')) and isinstance(model.lstm, nn.LSTM):
            self._prune_lstm(model, prune_percent, mode)
        else:
            self._prune_mlp(model, prune_percent, mode)

    def _prune_lstm(self, model, prune_percent, mode="FULL", ):
        lstm = model.lstm if hasattr(model, 'lstm') else model
        
        for name, param in lstm.named_parameters():
            if 'weight' in name:
                if mode == "FULL" or (mode == "IH" and "weight_ih" in name) or (mode == "HH" and "weight_hh" in name):
                    
                    mask = self.compute_mask(param, prune_percent)
                    param.data.mul_(mask)
                    if mode == "IH+HH":
                        mask = self.compute_mask(param, prune_percent)
                        param.data.mul_(mask)
                param.data[param.data == 0] = 0


    def _prune_mlp(self, model, prune_percent, mode="FULL"):
        layers = []
        #Get
        for module in model.modules():
            if isinstance(module, nn.Linear):
                layers.append(module)
        
        if not layers:
            return

        if mode == "FULL":
            all_weights = torch.cat([layer.weight.view(-1) for layer in layers])
            global_threshold = torch.kthvalue(
                torch.abs(all_weights),
                int(prune_percent / 100 * all_weights.numel())
            ).values
            
            for layer in layers:
                mask = (torch.abs(layer.weight) > global_threshold).float()
                layer.weight.data.mul_(mask)
                
        elif mode == "IH" and len(layers) >= 1:
            mask = self.compute_mask(layers[0].weight, prune_percent)
            layers[0].weight.data.mul_(mask)
            
        elif mode == "HH" and len(layers) > 2:
            for layer in layers[1:-1]:
                mask = self.compute_mask(layer.weight, prune_percent)
                layer.weight.data.mul_(mask)
                
        elif mode == "HO" and len(layers) >= 1:
            mask = self.compute_mask(layers[-1].weight, prune_percent)
            layers[-1].weight.data.mul_(mask)


class MagnitudePruner(BasePruner):
    def compute_mask(self, weight_tensor, prune_percent):
        weight_np = weight_tensor.cpu().detach().numpy().flatten()
        threshold = np.percentile(np.abs(weight_np), prune_percent)
        mask = (torch.abs(weight_tensor) > threshold).float()
        return mask

class RandomPruner(BasePruner):
    def compute_mask(self, weight_tensor, prune_percent):
        mask = torch.rand_like(weight_tensor) > (prune_percent / 100)
        return mask.float()

class L1Pruner(BasePruner):
    
    def compute_mask(self, weight_tensor, prune_percent):
        threshold = torch.kthvalue(torch.abs(weight_tensor.flatten()), int(prune_percent / 100 * weight_tensor.numel())).values
        mask = (torch.abs(weight_tensor) > threshold).float()
        return mask

class L2Pruner(BasePruner):
    def compute_mask(self, weight_tensor, prune_percent):
        norm = torch.norm(weight_tensor, p=2)
        threshold = norm * (prune_percent / 100)
        mask = (torch.abs(weight_tensor) > threshold).float()
        return mask


class PrunerThread(QThread):
    finished = pyqtSignal(bool)
    progress = pyqtSignal(str)
    progress_message = pyqtSignal(str)
    validation_info = pyqtSignal(dict)
    results_ready = pyqtSignal(dict)

    
    def __init__(self, model, prune_ratio, mode="FULL"):
        super().__init__()
        self.model = model
        self.prune_ratio = prune_ratio
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def run(self):
        try:
            if not hasattr(self, 'model'):
                raise AttributeError("Model attribute missing")
                
            if isinstance(self.model, str):
                raise ValueError(f"Model is still a string ('{self.model}')")
                
            if self.model is None:
                raise ValueError("Model is None")
                
            try:
                first_param = next(self.model.parameters(), None)
                if first_param is None:
                    raise RuntimeError("Model exists but has no parameters")
            except Exception as e:
                raise RuntimeError(f"Parameter access failed: {str(e)}") from e
            
            validation_data = {
                'model_type': type(self.model).__name__,
                'device': str(next(self.model.parameters()).device),
                'parameter_tensors': sum(1 for _ in self.model.parameters())
            }
            self.validation_info.emit(validation_data)
            
            self.progress_message.emit(f"\nApplying {self.mode} pruning at {self.prune_ratio:.0%} ratio")
            
            pruner = MagnitudePruner()
            pruner.apply_pruning(self.model, self.prune_ratio * 100, mode=self.mode)
            
            total_params = sum(p.numel() for p in self.model.parameters())
            zero_params = sum((p == 0).sum().item() for p in self.model.parameters())
            actual_sparsity = zero_params / total_params if total_params > 0 else 0
            
            results = {
                'total_parameters': total_params,
                'global_sparsity': actual_sparsity,
                'prune_mode': self.mode,
                'target_sparsity': self.prune_ratio,
                'actual_sparsity': actual_sparsity
            }
            self.results_ready.emit(results)
            
            self.finished.emit(True)
            
        except Exception as e:
            self.progress_message.emit(f"Pruning failed: {str(e)}")
            self.finished.emit(False)