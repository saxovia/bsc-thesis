import numpy as np



class PruningMethod:
    """Base class for pruning methods."""
    def prune(self, model, *args, **kwargs):
        """Apply pruning to the model."""
        raise NotImplementedError("Must be implemented in subclasses")

class MagnitudePruning(PruningMethod):
    def prune(self, model, pruning_rate=0.5):
        """Prunes smallest magnitude weights."""
        for layer in model.layers:
            threshold = np.percentile(np.abs(layer.weights), pruning_rate * 100)
            layer.weights[np.abs(layer.weights) < threshold] = 0
        return model

class RandomPruning(PruningMethod):
    def prune(self, model, pruning_rate=0.5):
        """Randomly removes weights."""
        for layer in model.layers:
            mask = np.random.rand(*layer.weights.shape) > pruning_rate
            layer.weights *= mask
        return model
