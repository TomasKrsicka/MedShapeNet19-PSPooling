# Trainer based on the examples from: https://lightning.ai/docs/pytorch/stable/starter/introduction.html (Apache License) and https://docs.wandb.ai/guides/integrations/lightning/ (MIT License)
# License available in the licenses folder

import argparse
import torch
import torch.nn.functional as F
import seaborn as sns
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import wandb
import os
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from graph_data_module import GraphDataset
from graph_data_module_structural import GraphDatasetStructural, split_dataset
from models import GSP_Classifier, GPSP_Classifier
from correct_precomp_data import process_global_matrices
from torchmetrics import Accuracy, ConfusionMatrix, Precision, Recall, F1Score
import sys
from utils import str2bool

class Mesh_Classifier(LightningModule):
    """
    PyTorch Lightning module for training a GNN-based classifier on mesh-structured data.

    Supports both structural (PS) pooling and selection-based pooling variants.

    Args:
        dataset_path (str): Path to the root dataset folder.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Training and validation batch size.
        is_relu (bool): Use ReLU activation if True, ELU otherwise.
        hidden_channels (int): Width of the hidden layers.
        pooling_vals (List[float]): Channel expansion values for pooling layers.
        use_ps_pooling (bool): Use PS pooling if True, selection pooling otherwise.
        wandb_logger (WandbLogger, optional): Weights & Biases logger for experiment tracking.
        num_classes (int): Number of classes for classification.
    """
    def __init__(self, dataset_path, learning_rate, batch_size,
                 is_relu, hidden_channels, pooling_vals, use_ps_pooling, wandb_logger=None, num_classes=2):
        super(Mesh_Classifier, self).__init__()
        self.save_hyperparameters()
        self.val_outputs = []
        self.val_labels = []
        self.model = None
        self.model_params = [6, hidden_channels, pooling_vals, is_relu, num_classes]
        self.wandb_logger = wandb_logger
        dataset = GraphDatasetStructural(root=dataset_path) if use_ps_pooling else GraphDataset(root=dataset_path)
        self.train_dataset, self.val_dataset, self.test_dataset = split_dataset(dataset)
        self.counter = 0
        self.label_dict = dataset.label_dict
        self.use_ps_pooling = use_ps_pooling
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.conf_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.precision = Precision(task='multiclass', num_classes=num_classes)
        self.recall = Recall(task='multiclass', num_classes=num_classes)
        self.f1 = F1Score(task='multiclass', num_classes=num_classes)

    def train_dataloader(self):
        """
        Returns the DataLoader for the training split.

        Returns:
            torch_geometric.loader.DataLoader: Training data loader.
        """
        return DataLoader(self.train_dataset, self.hparams.batch_size,
                          num_workers=4, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation split.

        Returns:
            torch_geometric.loader.DataLoader: Validation data loader.
        """
        return DataLoader(self.val_dataset, self.hparams.batch_size,
                          num_workers=4, persistent_workers=True, shuffle=False)

    def test_dataloader(self):
        """
        Returns the DataLoader for the test split.

        Returns:
            torch_geometric.loader.DataLoader: Test data loader.
        """
        return DataLoader(self.test_dataset, 2, shuffle=False)

    def setup(self, stage):
        """
        Initializes the classifier architecture based on pooling strategy.

        Args:
            stage (str): Optional. Unused here but required by Lightning.
        """
        self.model = GSP_Classifier(*self.model_params, 'cuda') if not self.use_ps_pooling \
            else GPSP_Classifier(*self.model_params, 'cuda')

    def forward(self, data):
        """
        Forward pass through the model. Switches logic depending on pooling type.

        Args:
            data (torch_geometric.data.Batch): Input batched graph data.

        Returns:
            torch.Tensor: Raw prediction logits.
        """
        if not self.use_ps_pooling:
            return self.model(data.x, data.edge_index, data.batch)
        index_matrices, weight_matrices, batch_tensors, pooled_edge_indices = process_global_matrices(data)
        return self.model(data.x, data.edge_index, data.batch, index_matrices,
                          weight_matrices, pooled_edge_indices, batch_tensors)


    def training_step(self, data, batch_idx):
        """
        Executes a single training step, computes loss and accuracy.

        Args:
            data (torch_geometric.data.Batch): A batch of training graphs.
            batch_idx (int): Index of the batch (unused).

        Returns:
            torch.Tensor: Computed training loss.
        """
        node_features = data.x
        batch = data.batch
        out = self(data)

        loss = F.cross_entropy(out, data.y)
        acc = self.train_acc(out, data.y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log('train_acc', acc, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, data, batch_idx):
        """
        Performs one step of validation, records predictions and logs metrics.

        Args:
            data (torch_geometric.data.Batch): A batch of validation graphs.
            batch_idx (int): Index of the batch (unused).

        Returns:
            torch.Tensor: Validation loss for the batch.
        """
        node_features = data.x
        batch = data.batch
        out = self(data)

        loss = F.cross_entropy(out, data.y)
        acc = self.val_acc(out, data.y)
        self.val_outputs.append(out)
        self.val_labels.append(data.y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)

        return loss
    def on_validation_epoch_end(self):
        """
        At the end of validation epoch:
        - Computes and logs precision, recall, F1.
        - Logs a confusion matrix to W&B (if enabled).
        - Saves a local confusion matrix plot.
        """
        all_outputs = torch.cat(self.val_outputs, dim=0)
        all_labels = torch.cat(self.val_labels, dim=0)
        conf_matrix = self.conf_matrix(all_outputs, all_labels).cpu().numpy()
        precision = self.precision(all_outputs, all_labels)
        recall = self.recall(all_outputs, all_labels)
        f1_score = self.f1(all_outputs, all_labels)
        xticks = [self.label_dict[i] for i in range(self.hparams.num_classes)]
        yticks = xticks

        self.log('val_precision', precision, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log('val_recall', recall, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log('val_f1_score', f1_score, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)

        plt.figure(figsize=(20, 14))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=xticks, yticklabels=yticks)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - Epoch {self.current_epoch + 1}')

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        confusion_matrix_img_path = f'confusion_matrix_epoch_{self.current_epoch + 1}.png'
        plt.savefig(confusion_matrix_img_path)
        plt.close()

        if self.wandb_logger:
            self.wandb_logger.experiment.log({
                "conf_matrix": wandb.Image(confusion_matrix_img_path)
            })

        self.val_outputs = []
        self.val_labels = []

    def configure_optimizers(self):
        """
        Configures the Adam optimizer.

        Returns:
            torch.optim.Optimizer: Configured optimizer instance.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


def t_train(config=None):
    """
    Trains the mesh classifier using parameters defined in a config dict.

    Args:
        config (dict): Configuration dictionary for model setup and training.
    """
    with wandb.init(config=config):
        wandb_logger = WandbLogger(log_model=True)
        config = wandb.config

        model = Mesh_Classifier(
            batch_size=config.batch_size,
            dataset_path=config.dataset_path,
            learning_rate=config.learning_rate,
            is_relu=config.is_relu,
            hidden_channels=config.hidden_channels,
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
            patience=50,  
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
    Main entry point: parses arguments, initializes W&B, sets up training and launches the run.
    """
    parser = argparse.ArgumentParser(description='Configure training parameters.')
    parser.add_argument('--use_ps_pooling', type=str2bool, default=False, help='Use PS pooling if True, SAG pooling if False')
    parser.add_argument('--is_relu', type=str2bool, default=False, help='Use relu activation (if false, elu)')
    parser.add_argument('--pooling_vals', type=str, default="1 2 4", help='channel expansion')
    parser.add_argument('--hidden_channels', type=int, default=64, help='number of hidden channels after outer mlp')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--run_name', type=str)

    args = parser.parse_args()
    return args


def main():
    os.environ["WANDB_MODE"] = "offline"
    torch.set_float32_matmul_precision('medium')
    sys.path.append("..")
    args = parse_args()

    # Define the sweep configuration
    config = {
        'learning_rate': args.learning_rate,
        'is_relu': args.is_relu,
        'pooling_vals': list(map(float, args.pooling_vals.split())),
        'hidden_channels': args.hidden_channels,
        'use_ps_pooling': args.use_ps_pooling,
        'epochs': 1000,
        'batch_size': args.batch_size,
        'dataset_path': 'dataset/',
    }

    wandb.init(project='mesh_classifier', config=config, name=args.run_name)
    t_train(config)
    wandb.finish()


if __name__ == "__main__":
    main()

