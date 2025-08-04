import torch.nn as nn



class PrecomputedStructuralPooling(nn.Module):
    """
    Precomputed structural pooling layer using static vertex indices.
    Aggregates features from selected vertices using max pooling.
    """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        
    def forward(self, x, indices, weights):
        selected_features = x[indices]
        return selected_features.max(dim=1).values

class PrecomputedStructuralUnpooling(nn.Module):
    """
    Precomputed structural unpooling layer using static vertex indices and weights.
    Redistributes pooled features back to original resolution using weighted sums.
    """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
    def forward(self, x, indices, weights):
        device = x.device
        weights = weights.unsqueeze(-1)
        selected_features = x[indices]
        weighted_features = selected_features * weights
        out_features = weighted_features.sum(dim=1, dtype=x.dtype)
        return out_features
