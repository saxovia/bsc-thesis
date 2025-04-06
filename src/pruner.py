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
        if isinstance(model, LSTMNet):
            for name, param in model.named_parameters():
                if "weight" in name:  # Ignore biases
                    if mode == "FULL" or (mode == "IH" and "weight_ih" in name) or (mode == "HH" and "weight_hh" in name):
                        mask = self.compute_mask(param, prune_percent)
                        param.data.mul_(mask)
                    elif mode == "HTO" and hasattr(model, "fc") and "fc.weight" in name:
                        mask = self.compute_mask(param, prune_percent)
                        param.data.mul_(mask)
        elif isinstance(model, MLPNet) or isinstance(model, nn.Sequential):
            layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
            if mode == "FULL":
                for layer in layers:
                    mask = self.compute_mask(layer.weight, prune_percent)
                    layer.weight.data.mul_(mask)
            elif mode == "IH" and len(layers) > 0:
                mask = self.compute_mask(layers[0].weight, prune_percent)
                layers[0].weight.data.mul_(mask)
            elif mode == "HH" and len(layers) > 2:
                for layer in layers[1:-1]:
                    mask = self.compute_mask(layer.weight, prune_percent)
                    layer.weight.data.mul_(mask)
            elif mode == "HTO" and len(layers) > 0:
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
