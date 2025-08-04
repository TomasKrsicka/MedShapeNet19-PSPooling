import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import MLP, GATv2Conv, AttentionalAggregation, SAGPooling
from PrecomputedStructuralPooling import PrecomputedStructuralPooling
from torch_geometric.nn.models import GAT
import torch.nn.functional as F

class GSP_Autoencoder(nn.Module):
    """
    Autoencoder using selection-based pooling (SAGPooling) for hierarchical graph representation learning.

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Number of hidden channels before first pooling stage.
        use_global_pooling (bool): Whether to apply global pooling after encoding.
        pooling_vals (List[float]): List of feature scaling factors.
        is_relu (bool): If True, uses ReLU activation; otherwise uses ELU.
        out_channels (int or None): Optional override for output feature size.
        device (str or torch.device): Device to run the model on.
    """
    class Encoder(nn.Module):
        """
        Encoder with GAT-based feature extraction and SAGPooling for resolution reduction.

        Args:
            device (str): Device to run the model on.
            in_channels (int): Number of input node features.
            hidden_channels (int): Number of hidden channels before first pooling stage.
            is_relu (bool): If True, uses ReLU activation; otherwise uses ELU.
            pooling_vals (List[float]): List of feature scaling factors.
            use_global_pooling (bool): Whether to apply global pooling after encoding.
        """
        def __init__(self, device, in_channels, hidden_channels, is_relu,
                     pooling_vals, use_global_pooling=True):
            super().__init__()
            self.use_global_pooling = use_global_pooling
            self.activation = F.relu if is_relu else F.elu
            self.in_layer = GAT(in_channels, hidden_channels, out_channels=None, num_layers=4, v2=True)
            self.gcns = []
            for i in range(len(pooling_vals) - 1):
                self.gcns.append(GATv2Conv(int(hidden_channels * pooling_vals[i]),
                                           int(hidden_channels * pooling_vals[i + 1])).to(device))
            self.pools = []
            for v in pooling_vals[:-1]:
                self.pools.append(SAGPooling(int(hidden_channels * v), ratio=0.7).to(device))
            in_gcn_channels = int(hidden_channels * pooling_vals[-1])
            self.att = AttentionalAggregation(Linear(int(in_gcn_channels), 1)).to(device)
            self.last_linear = Linear(int(in_gcn_channels), 2)

        def forward(self, x, edge_index, batch):
            x = self.activation(self.in_layer(x, edge_index))
            unpool_infos = []
            for gcn, pool in zip(self.gcns, self.pools):
                x_shape = x.shape
                x, nedge_index, _, batch, perm, _ = pool(x, edge_index, None, batch)
                unpool_infos.append((perm, x_shape, edge_index))
                edge_index = nedge_index
                x = self.activation(gcn(x, edge_index))
            per_node_embedding = self.last_linear(x)
            x = x if not self.use_global_pooling else self.att(x, batch)
            return x, edge_index, batch, unpool_infos, per_node_embedding

    class Decoder(nn.Module):
        """
        Decoder that reconstructs graph features using stored unpooling indices and GAT layers.

        Args:
            device (str): Device to run the model on.
            hidden_channels (int): Number of hidden channels after the last unpooling stage.
            out_channels (int): Final output feature dimension.
            is_relu (bool): If True, uses ReLU activation; otherwise uses ELU.
            pooling_vals (List[float]): Reversed list of feature scaling factors.
            use_global_pooling (bool): If True, expects pooled bottleneck vector.
        """
        def __init__(self, device, hidden_channels, out_channels, is_relu, pooling_vals,
                     use_global_pooling=True):
            super().__init__()
            self.activation = F.relu if is_relu else F.elu
            self.device = device
            in_gcn_channels = int(hidden_channels * pooling_vals[0])
            self.mlp = MLP([hidden_channels, hidden_channels // 2, out_channels]).to(device)
            self.out_preprocess = GATv2Conv(hidden_channels, hidden_channels).to(device)
            self.unpool_mlp = MLP([in_gcn_channels + 2, in_gcn_channels]).to(device)
            self.use_global_pooling = use_global_pooling
            self.gcns = []
            for i in range(len(pooling_vals) - 1):
                self.gcns.append(GATv2Conv(int(hidden_channels * pooling_vals[i]),
                                           int(hidden_channels * pooling_vals[i + 1])).to(device))

        def forward(self, x, edge_index, unpool_infos, pooled_batch, original_batch):
            x = x if not self.use_global_pooling else self.unpool_mlp(x, pooled_batch)
            for gcn, (perm, unpool_shape, nedge_index) in zip(self.gcns, unpool_infos):
                x = self.activation(gcn(x, edge_index))
                x_modified = torch.zeros(unpool_shape, device=self.device)
                x_modified[perm] = x
                mask = torch.ones(unpool_shape[0], dtype=torch.bool, device=self.device)
                mask[perm] = False
                edge_index = nedge_index
                x = x_modified
            x = self.out_preprocess(x, edge_index)
            x = self.mlp(x, original_batch)

            return x, edge_index

    def __init__(self, in_channels, hidden_channels, use_global_pooling, pooling_vals,
                 is_relu=False, out_channels=None, device=None):
        device = device if device else 'cuda'
        super().__init__()
        self.encoder = self.Encoder(device, in_channels, hidden_channels, is_relu,
                                    pooling_vals, use_global_pooling).to(device)
        self.decoder = self.Decoder(device, hidden_channels, in_channels if not out_channels else out_channels,
                                    is_relu, pooling_vals[::-1], use_global_pooling).to(device)
        self.use_global_pooling = use_global_pooling

    def forward(self, x, edge_index, batch):
        nx, nedge_index, batch2, unpool_info, per_node_embedding = self.encoder(x, edge_index, batch)

        nx = nx if not self.use_global_pooling else torch.cat([per_node_embedding, nx[batch2]], dim=1)

        rx, redge_index = self.decoder(nx, nedge_index, unpool_info[::-1], batch2, batch)
        return rx, redge_index
    def infer(self, x, edge_index, batch):
        nx, _, _, _, _ = self.encoder(x, edge_index, batch)
        return nx

class GPSP_Autoencoder(nn.Module):
    """
    Autoencoder using precomputed structural pooling for hierarchical graph representation learning.

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Number of hidden channels before the first pooling stage.
        use_global_pooling (bool): Whether to apply global pooling after encoding.
        pooling_vals (List[float]): List of feature scaling factors for each pooling level.
        is_relu (bool): If True, uses ReLU activation; otherwise uses ELU.
        out_channels (int or None): Optional override for output feature size.
        device (str or torch.device): Device to run the model on.
    """
    class Encoder(nn.Module):
        """
        Encoder with GAT-based feature extraction and precomputed structural pooling for resolution reduction.

        Args:
            device (str or torch.device): Device to run the model on.
            in_channels (int): Number of input node features.
            hidden_channels (int): Number of hidden channels before the first pooling stage.
            is_relu (bool): If True, uses ReLU activation; otherwise uses ELU.
            pooling_vals (List[float]): List of feature scaling factors for each pooling level.
            use_global_pooling (bool): Whether to apply global pooling after encoding.
        """
        def __init__(self, device, in_channels, hidden_channels, is_relu,
                     pooling_vals,  use_global_pooling=True):
            super().__init__()
            self.use_global_pooling = use_global_pooling
            self.activation = F.relu if is_relu else F.elu
            self.in_layer = GAT(in_channels, hidden_channels, out_channels=None, num_layers=4, v2=True)
            self.gcns = []
            for i in range(len(pooling_vals) - 1):
                self.gcns.append(GATv2Conv(int(hidden_channels * pooling_vals[i]),
                                           int(hidden_channels * pooling_vals[i + 1])).to(device))
            self.pools = []
            for v in pooling_vals[:-1]:
                self.pools.append(PrecomputedStructuralPooling(int(hidden_channels * v)).to(device))
            in_gcn_channels = int(hidden_channels * pooling_vals[-1])
            self.att = AttentionalAggregation(Linear(int(in_gcn_channels), 1)).to(device)
            self.last_linear = Linear(int(in_gcn_channels), 2)

        def forward(self, x, edge_index, index_matrices, weight_matrices, pooled_edge_indices, last_batch):
            x = self.activation(self.in_layer(x, edge_index))
            for i, (gcn, pool) in enumerate(zip(self.gcns, self.pools)):
                indices = index_matrices[i]
                weights = weight_matrices[i]
                x = pool(x, indices, weights)
                edge_index = pooled_edge_indices[i]
                x = self.activation(gcn(x, edge_index))
            per_node_embedding = self.last_linear(x)
            x = x if not self.use_global_pooling else self.att(x, last_batch)
            return x, edge_index, last_batch, per_node_embedding

    class Decoder(nn.Module):
        """
        Decoder that reconstructs graph features using precomputed unpooling operations and GAT layers.

        Args:
            device (str or torch.device): Device to run the model on.
            hidden_channels (int): Number of hidden channels after the last unpooling stage.
            out_channels (int): Final output feature dimension.
            is_relu (bool): If True, uses ReLU activation; otherwise uses ELU.
            pooling_vals (List[float]): Reversed list of feature scaling factors.
            use_global_pooling (bool): If True, expects pooled bottleneck vector as decoder input.
        """
        def __init__(self, device, hidden_channels, out_channels, is_relu, pooling_vals,
                     use_global_pooling=True):
            super().__init__()
            self.activation = F.relu if is_relu else F.elu
            self.device = device
            in_gcn_channels = int(hidden_channels * pooling_vals[0])
            self.mlp = MLP([hidden_channels, hidden_channels // 2, out_channels]).to(device)
            self.out_preprocess = GATv2Conv(hidden_channels, hidden_channels).to(device)
            self.unpool_mlp = MLP([in_gcn_channels + 2, in_gcn_channels]).to(device)
            self.use_global_pooling = use_global_pooling
            self.gcns = []
            for i in range(len(pooling_vals) - 1):
                self.gcns.append(GATv2Conv(int(hidden_channels * pooling_vals[i]),
                                           int(hidden_channels * pooling_vals[i + 1])).to(device))
            self.unpools = []
            for v in pooling_vals[1:]:
                self.unpools.append(PrecomputedStructuralPooling(int(hidden_channels * v)).to(device))

        def forward(self, x, original_edge_index, index_matrices, weight_matrices, pooled_edge_indices,
                    pooled_batch, original_batch):
            x = x if not self.use_global_pooling else self.unpool_mlp(x, pooled_batch)
            edge_index = pooled_edge_indices[0]
            for i, (gcn, unpool) in enumerate(zip(self.gcns, self.unpools)):
                x = self.activation(gcn(x, edge_index))
                indices = index_matrices[i]
                weights = weight_matrices[i]
                x = unpool(x, indices, weights)
                edge_index = pooled_edge_indices[min(i, len(pooled_edge_indices)-1)]
            edge_index = original_edge_index
            x = self.out_preprocess(x, edge_index)
            x = self.mlp(x, original_batch)

            return x, edge_index

    def __init__(self, in_channels, hidden_channels, use_global_pooling, pooling_vals,
                 is_relu=False, out_channels=None, device=None):
        device = device if device else 'cuda'
        self.ps = len(pooling_vals)-1
        super().__init__()
        self.encoder = self.Encoder(device, in_channels, hidden_channels, is_relu,
                                    pooling_vals, use_global_pooling).to(device)
        self.decoder = self.Decoder(device, hidden_channels, in_channels if not out_channels else out_channels,
                                    is_relu, pooling_vals[::-1],  use_global_pooling).to(device)
        self.use_global_pooling = use_global_pooling

    def forward(self, x, edge_index, batch, index_matrices, weight_matrices, pooled_edge_indices, batches):
        r_ps = len(index_matrices) - self.ps
        nx, nedge_index, pooled_batch, per_node_embedding = self.encoder(x, edge_index, index_matrices[:self.ps],
                                                                        weight_matrices[:self.ps], pooled_edge_indices[:self.ps],
                                                                        batches[self.ps-1])
        nx = nx if not self.use_global_pooling else torch.cat([per_node_embedding, nx[pooled_batch]], dim=1)

        rx, redge_index = self.decoder(nx, edge_index, index_matrices[r_ps:], weight_matrices[r_ps:],
                                       pooled_edge_indices[:self.ps][::-1], pooled_batch, batch)
        return rx, redge_index

    def infer(self, x, edge_index, batch, index_matrices, weight_matrices, pooled_edge_indices, batches):
        nx, _, _, _ = self.encoder(x, edge_index, index_matrices[:self.ps], weight_matrices[:self.ps],
                                                                         pooled_edge_indices[:self.ps],
                                                                         batches[self.ps - 1])
        return nx
class GSP_Classifier(nn.Module):
    """
    Graph classifier using SAGPooling and GAT-based hierarchical feature extraction.

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Number of hidden channels before the first pooling stage.
        pooling_vals (List[float]): List of feature scaling factors for each pooling level.
        is_relu (bool): If True, uses ReLU activation; otherwise uses ELU.
        device (str or torch.device): Device to run the model on.
        num_classes (int): Number of target classes for classification.
    """
    class Encoder(nn.Module):
        """
        Encoder for classification using GAT-based feature extraction and SAGPooling for graph resolution reduction.

        Args:
            device (str or torch.device): Device to run the model on.
            in_channels (int): Number of input node features.
            hidden_channels (int): Number of hidden channels before the first pooling stage.
            is_relu (bool): If True, uses ReLU activation; otherwise uses ELU.
            pooling_vals (List[float]): List of feature scaling factors for each pooling level.
            num_classes (int): Number of output classes for classification.
        """
        def __init__(self, device, in_channels, hidden_channels, is_relu,
                     pooling_vals, num_classes=19):
            super().__init__()
            self.activation = F.relu if is_relu else F.elu
            self.in_layer = GAT(in_channels, hidden_channels, out_channels=None, num_layers=4, v2=True)
            self.gcns = []
            for i in range(len(pooling_vals) - 1):
                self.gcns.append(GATv2Conv(int(hidden_channels * pooling_vals[i]),
                                           int(hidden_channels * pooling_vals[i + 1])).to(device))
            self.pools = []
            for v in pooling_vals[:-1]:
                self.pools.append(SAGPooling(int(hidden_channels * v), ratio=0.7).to(device))
            in_gcn_channels = int(hidden_channels * pooling_vals[-1])
            self.att = AttentionalAggregation(Linear(int(in_gcn_channels), 1)).to(device)
            self.classifier = Linear(int(hidden_channels * pooling_vals[-1]), num_classes).to(device)

        def forward(self, x, edge_index, batch):
            x = self.activation(self.in_layer(x, edge_index))
            for gcn, pool in zip(self.gcns, self.pools):
                x, nedge_index, _, batch, perm, _ = pool(x, edge_index, None, batch)
                edge_index = nedge_index
                x = self.activation(gcn(x, edge_index))
            x = self.att(x, batch)
            x = self.classifier(x)
            return x

    def __init__(self, in_channels, hidden_channels, pooling_vals,
                 is_relu=False, num_classes=19, device=None):
        device = device if device else 'cuda'
        super().__init__()
        self.encoder = self.Encoder(device, in_channels, hidden_channels, is_relu,
                                    pooling_vals, num_classes).to(device)

    def forward(self, x, edge_index, batch):
        nx = self.encoder(x, edge_index, batch)
        return nx

class GPSP_Classifier(nn.Module):
    """
    Graph classifier using precomputed structural pooling and GAT-based hierarchical feature extraction.

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Number of hidden channels before the first pooling stage.
        pooling_vals (List[float]): List of feature scaling factors for each pooling level.
        is_relu (bool): If True, uses ReLU activation; otherwise uses ELU.
        device (str or torch.device): Device to run the model on.
        num_classes (int): Number of target classes for classification.
    """
    class Encoder(nn.Module):
        """
        Encoder for classification using GAT-based feature extraction and precomputed structural pooling.

        Args:
            device (str or torch.device): Device to run the model on.
            in_channels (int): Number of input node features.
            hidden_channels (int): Number of hidden channels before the first pooling stage.
            is_relu (bool): If True, uses ReLU activation; otherwise uses ELU.
            pooling_vals (List[float]): List of feature scaling factors for each pooling level.
            num_classes (int): Number of output classes for classification.
        """
        def __init__(self, device, in_channels, hidden_channels, is_relu,
                     pooling_vals, num_classes=19):
            super().__init__()
            self.activation = F.relu if is_relu else F.elu
            self.in_layer = GAT(in_channels, hidden_channels, out_channels=None, num_layers=4, v2=True)
            self.gcns = []
            for i in range(len(pooling_vals) - 1):
                self.gcns.append(GATv2Conv(int(hidden_channels * pooling_vals[i]),
                                           int(hidden_channels * pooling_vals[i + 1])).to(device))
            self.pools = []
            for v in pooling_vals[:-1]:
                self.pools.append(PrecomputedStructuralPooling(int(hidden_channels * v)).to(device))
            in_gcn_channels = int(hidden_channels * pooling_vals[-1])
            self.att = AttentionalAggregation(Linear(int(in_gcn_channels), 1)).to(device)
            self.classifier = torch.nn.Linear(int(hidden_channels * pooling_vals[-1]), num_classes).to(device)

        def forward(self, x, edge_index, index_matrices, weight_matrices, pooled_edge_indices, last_batch):
            x = self.activation(self.in_layer(x, edge_index))
            for i, (gcn, pool) in enumerate(zip(self.gcns, self.pools)):
                indices = index_matrices[i]
                weights = weight_matrices[i]
                x = pool(x, indices, weights)
                edge_index = pooled_edge_indices[i]
                x = self.activation(gcn(x, edge_index))
            x = self.att(x, last_batch)
            x = self.classifier(x)
            return x


    def __init__(self, in_channels, hidden_channels, pooling_vals,
                 is_relu=False, num_classes=19, device=None):
        device = device if device else 'cuda'
        self.ps = len(pooling_vals)-1
        super().__init__()
        self.encoder = self.Encoder(device, in_channels, hidden_channels, is_relu,
                                    pooling_vals, num_classes).to(device)

    def forward(self, x, edge_index, batch, index_matrices, weight_matrices, pooled_edge_indices, batches):
        nx = self.encoder(x, edge_index, index_matrices[:self.ps],
                          weight_matrices[:self.ps], pooled_edge_indices,
                            batches[self.ps-1])
        return nx
