# Trainer based on the examples from: https://lightning.ai/docs/pytorch/stable/starter/introduction.html (Apache License) and https://docs.wandb.ai/guides/integrations/lightning/ (MIT License)
# 3D loss computation based on example from: https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh (BSD License)
# License available in the licenses folder

import argparse
import torch
import os
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from graph_data_module_structural import GraphDatasetStructural, split_dataset
from graph_data_module import GraphDataset
from models import GSP_Autoencoder, GPSP_Autoencoder
from correct_precomp_data import process_global_matrices
import sys
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import IO
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from utils import str2bool

class Mesh_Autoencoder(LightningModule):
    """
    PyTorch Lightning module for training a mesh autoencoder using either 
    structural (PS) pooling or selection-based pooling.

    Args:
        dataset_path (str): Path to the dataset directory.
        learning_rate (float): Learning rate for optimizer.
        batch_size (int): Batch size used in training and validation.
        regularize_loss (bool): Whether to include additional mesh regularization terms in the loss.
        is_relu (bool): If True, use ReLU activations. If False, use ELU.
        hidden_channels (int): Hidden channel width for internal layers.
        use_global_pooling (bool): Whether to use global pooling in the bottleneck.
        pooling_vals (List[float]): List of expansion factors for the pooling structure.
        use_ps_pooling (bool): Use PS pooling if True, otherwise selection pooling.
        wandb_logger (WandbLogger, optional): Optional W&B logger for experiment tracking.
    """
    def __init__(self, dataset_path, learning_rate, batch_size, regularize_loss,
                 is_relu, hidden_channels, use_global_pooling, pooling_vals, use_ps_pooling, wandb_logger=None):
        super(Mesh_Autoencoder, self).__init__()
        self.save_hyperparameters()
        self.val_outputs = []
        self.val_labels = []
        self.model = None
        self.model_params = [6, hidden_channels, use_global_pooling, pooling_vals, is_relu, 3]
        self.regularize_loss = regularize_loss
        self.wandb_logger = wandb_logger
        dataset = GraphDatasetStructural(root=dataset_path) if use_ps_pooling else GraphDataset(root=dataset_path)
        self.train_dataset, self.val_dataset, self.test_dataset = split_dataset(dataset)
        self.use_ps_pooling = use_ps_pooling
        self.counter = 0

    def train_dataloader(self):
        """
        Returns the DataLoader for the training split.
        Uses persistent workers and shuffles the dataset.

        Returns:
            torch_geometric.loader.DataLoader: The DataLoader for training.
        """
        return DataLoader(self.train_dataset, self.hparams.batch_size,
                          num_workers=7, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation split.
        Workers are persistent but no shuffling is applied.

        Returns:
            torch_geometric.loader.DataLoader: The DataLoader for validation.
        """
        return DataLoader(self.val_dataset, self.hparams.batch_size,
                          num_workers=4, persistent_workers=True, shuffle=False)

    def test_dataloader(self):
        """
        Returns the DataLoader for the test split.

        Returns:
            torch_geometric.loader.DataLoader: The DataLoader for testing.
        """
        return DataLoader(self.test_dataset, 2, shuffle=False)

    def setup(self, stage):
        """
        Initializes the model architecture based on pooling strategy.
        Called once per training job.

        Args:
            stage (str): Optional. Not used explicitly here.
        """
        self.model = GPSP_Autoencoder(*self.model_params, 'cuda') if self.use_ps_pooling\
                            else GSP_Autoencoder(*self.model_params, 'cuda')

    def forward(self, data):
        """
        Forward pass through the model. Chooses the correct pipeline depending on pooling strategy.

        Args:
            data (torch_geometric.data.Batch): Input graph batch.

        Returns:
            tuple: Reconstructed node features and optionally reconstructed edge indices.
        """
        if not self.use_ps_pooling:
            return self.model(data.x, data.edge_index, data.batch)
        index_matrices, weight_matrices, batch_tensors, pooled_edge_indices = process_global_matrices(data)
        return self.model(data.x, data.edge_index, data.batch, index_matrices,
                          weight_matrices, pooled_edge_indices, batch_tensors)

    def normal_loss(self, v1, v2, f1, f2=None):
        """
        Computes Chamfer distance and additional regularization terms between predicted and target meshes.

        Args:
            v1 (List[Tensor]): List of vertex tensors for the target mesh.
            v2 (List[Tensor]): List of vertex tensors for the predicted mesh.
            f1 (List[Tensor]): List of face index tensors for the target mesh.
            f2 (List[Tensor], optional): Face indices for the predicted mesh. Defaults to `f1`.

        Returns:
            Tuple[float]: Chamfer distance, Hausdorff-like max CD, normal consistency, edge length loss, and Laplacian smoothing.
        """
        f2 = f2 if f2 else f1
        m1 = Meshes(verts=v1, faces=f1)
        m2 = Meshes(verts=v2, faces=f2)
        p1, n1 = sample_points_from_meshes(m1, 10000, return_normals=True)
        p2, n2 = sample_points_from_meshes(m2, 10000, return_normals=True)
        cd, _ = chamfer_distance(p1, p2)
        cdmax, _ = chamfer_distance(p1, p2, point_reduction='max')
        mnc = mesh_normal_consistency(m2)
        mel = mesh_edge_loss(m2, target_length=1)
        ll = mesh_laplacian_smoothing(m2, method="uniform")
        return cd, cdmax, mnc, mel, ll

    def split_nodes_by_graph(self, batch_tensor, node_tensor):
        """
        Splits batched node features into individual meshes based on batch index.

        Args:
            batch_tensor (Tensor): Batch indices.
            node_tensor (Tensor): Node features.

        Returns:
            List[Tensor]: List of node tensors for each graph in the batch.
        """
        return [node_tensor[batch_tensor == i] for i in range(batch_tensor.max().item() + 1)]

    def training_step(self, data, batch_idx):
        """
        Executes a single training step, computes reconstruction and loss.

        Args:
            data (torch_geometric.data.Batch): Input graph batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Scalar loss to be backpropagated.
        """
        node_features = data.x
        batch = data.batch
        reconstructed_x, r_edges = self(data)

        cd = mnc = mel = ll = cdmax = 0
        re_nodes = self.split_nodes_by_graph(batch, reconstructed_x)
        cd, cdmax, mnc, mel, ll = self.normal_loss(self.split_nodes_by_graph(batch, node_features[:, :3]), re_nodes,
                                                [torch.from_numpy(f).to(self.device) for f in data.faces])
        self.log('chamfer distance', cd, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log('Hausdorff distance', cdmax, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log('mesh normal consistency', mnc, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log('mesh edge loss', mel, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log('laplacian loss', ll, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        regularizers = 0.1 * mnc + 0.3 * mel + 0.3 * ll if self.regularize_loss else 0
        loss = cd 
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, data, batch_idx):
        """
        Performs a validation step and logs Chamfer distance.

        Args:
            data (torch_geometric.data.Batch): Input graph batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        node_features = data.x
        batch = data.batch
        reconstructed_x, r_edges = self(data)
        re_nodes = self.split_nodes_by_graph(batch, reconstructed_x)
        cd, _, _, _, _ = self.normal_loss(self.split_nodes_by_graph(batch, node_features[:, :3]), re_nodes,
                                          [torch.from_numpy(f).to(self.device) for f in data.faces])
        loss = cd
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        return loss

    def on_validation_epoch_end(self):
        """
        At the end of each validation epoch, saves reconstructed mesh examples every 20 epochs.
        """
        if self.current_epoch %20 == 0 or self.current_epoch < 2:
            tloader = self.test_dataloader()
            for i, data in enumerate(tloader):
                batch = data.batch
                reconstructed_x, r_edges = self(data.to(self.device))
                re_nodes = self.split_nodes_by_graph(batch, reconstructed_x)
                for i in range(len(data.faces)):
                    IO().save_mesh(
                        data=Meshes(verts=[re_nodes[i]], faces=[torch.from_numpy(data.faces[i]).to(self.device)]),
                        path=f"./plot_out/{self.current_epoch}_{data.y[i]}.obj")

    def configure_optimizers(self):
        """
        Configures the optimizer (Adam).

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


def t_train(config=None):
    """
    Trains the mesh autoencoder using configuration from WandB.
    Sets up the logger, callbacks, trainer, and initiates training.

    Args:
        config (dict): Configuration dictionary, passed from WandB sweep or script.
    """
    os.makedirs("./plot_out", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    with wandb.init(config=config):
        wandb_logger = WandbLogger(log_model=True)
        config = wandb.config
        model = Mesh_Autoencoder(
            batch_size=config.batch_size,
            dataset_path=config.dataset_path,
            learning_rate=config.learning_rate,
            regularize_loss=config.regularize_loss,
            is_relu=config.is_relu,
            hidden_channels=config.hidden_channels,
            use_global_pooling=config.use_global_pooling,
            pooling_vals=config.pooling_vals,
            use_ps_pooling=config.use_ps_pooling,
            wandb_logger=wandb_logger
        )
        checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",  
        filename="model-{epoch:02d}-{val_loss:.2f}",  
        save_top_k=-1, 
        save_weights_only=True,  
        every_n_epochs=1,  
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", 
            patience=100,  
            mode="min", 
            verbose=True,
        )
        trainer = pl.Trainer(
            max_epochs=config['epochs'],
            callbacks=[checkpoint_callback, early_stopping_callback],
            sync_batchnorm=True,
            logger=wandb_logger,
            devices=1,
            accelerator='gpu'
        )
        trainer.fit(model, model.train_dataloader(), model.val_dataloader())

def parse_args():
    """
    Parses command-line arguments for training.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Configure training parameters.')
    parser.add_argument('--regularize_loss', type=str2bool, default=True,
                        help='Add loss of model normals to global loss')
    parser.add_argument('--is_relu', type=str2bool, default=False, help='Use relu activation (if false, elu)')
    parser.add_argument('--use_global_pooling', type=str2bool, default=True, help='Is bottleneck 1 1D tensor?')
    parser.add_argument('--pooling_vals', type=str, default="1 2 4", help='channel expansion')
    parser.add_argument('--use_ps_pooling', type=str2bool, default=True, help='Use PS pooling if True, SAG pooling if False')
    parser.add_argument('--hidden_channels', type=int, default=64, help='number of hidden channels after outer mlp')
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=19)
    parser.add_argument('--run_name', type=str)

    args = parser.parse_args()
    return args


def main():
    """
    Main entry point: parses arguments, initializes W&B, sets up training and launches the run.
    """
    os.environ["WANDB_MODE"] = "offline" #comment out if api key has been set
    torch.set_float32_matmul_precision('medium')
    sys.path.append("..")
    args = parse_args()
    config = {
        'learning_rate': args.learning_rate,
        'is_relu': args.is_relu,
        'use_global_pooling': args.use_global_pooling,
        'pooling_vals': list(map(float, args.pooling_vals.split())),
        'hidden_channels': args.hidden_channels,
        'use_ps_pooling': args.use_ps_pooling,
        'epochs': 1000,
        'batch_size': args.batch_size,
        'regularize_loss': args.regularize_loss,
        'dataset_path': 'dataset/',
    }

    wandb.init(project='Mesh_Autoencoder', config=config, name=args.run_name)
    t_train(config)
    wandb.finish()


if __name__ == "__main__":
    main()

