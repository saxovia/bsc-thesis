import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from src.neuralnetwork import MLPNet, LSTMNet

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
