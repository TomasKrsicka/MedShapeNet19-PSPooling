import csv
import os
import torch_geometric
import numpy as np
import torch
import trimesh
from torch_geometric.data import Dataset, Data
from sklearn.model_selection import train_test_split


class GraphDataset(Dataset):
    """
    Custom PyTorch Geometric Dataset to handle loading mesh data from a folder-based class layout.

    Each class is represented by a subdirectory, and each sample is a .stl file.

    Args:
        root (str): Root directory containing subfolders for each class label.
        transform (callable, optional): Optional transform to apply on a Data object.
        pre_transform (callable, optional): Optional pre-transform applied before saving to disk.
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.graph_paths = []
        self.labels = []
        self.label_dict = {}
        class_to_idx = {}
        next_idx = 0
        self.split_ids = []  # 0 = train, 1 = val, 2 = test

        split_info_path = os.path.join(root, 'split_info.csv')
        assert os.path.isfile(split_info_path), f"split_info.csv not found at {split_info_path}"

        with open(split_info_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                mesh_path, split_idx = row
                mesh_path = mesh_path.strip()
                split_idx = int(split_idx.strip())
                full_path = os.path.normpath(mesh_path)

                class_name = os.path.basename(os.path.dirname(full_path))
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = next_idx
                    next_idx += 1
                label_idx = class_to_idx[class_name]

                self.graph_paths.append(mesh_path)
                self.labels.append(label_idx)
                self.split_ids.append(split_idx)
        self.label_dict = {v: k for k, v in class_to_idx.items()}

    def len(self):
        return len(self.graph_paths)

    def get(self, idx):
        features, edge_index, faces = extract_vertex_features_and_edges(self.graph_paths[idx])
        label = self.labels[idx]
        data = Data(x=torch.from_numpy(features.copy()),
                    edge_index=torch_geometric.utils.to_undirected(torch.from_numpy(edge_index.copy())),
                    y=torch.tensor([label], dtype=torch.long),
                    faces=faces,
                    path=self.graph_paths[idx])
        return data


def scale_and_shift_points(points, mean_edge_length):    
    """
    Normalizes mesh by centering and scaling to a target mean edge length.

    Args:
        points (ndarray): Array of 3D vertex coordinates.
        mean_edge_length (float): Mean edge length used to scale the mesh.

    Returns:
        ndarray: Scaled and centered 3D coordinates.
    """
    centroid = points.mean(axis=0)
    centered_points = points - centroid
    scaled_points = centered_points / mean_edge_length
    return scaled_points
def extract_vertex_features_and_edges(mesh_name):
    """
    Extracts per-vertex features and edge indices from a 3D mesh file.

    Features include scaled coordinates and vertex normals.

    Args:
        mesh_name (str): Path to the mesh file (expected .stl format).

    Returns:
        Tuple[ndarray, ndarray, ndarray]:
            - features (ndarray): Vertex features (coordinates + normals).
            - edge_index (ndarray): Edge indices as a 2-row array.
            - faces (ndarray): Triangular face indices of the mesh. Used for reconstruction
    """
    mesh = trimesh.load_mesh(mesh_name)
    return np.hstack([scale_and_shift_points(mesh.vertices, mesh.edges_unique_length.mean()), mesh.vertex_normals], dtype=np.float32), mesh.edges_unique.T, mesh.faces

def split_dataset(dataset):
    """
    Returns train, validation, and test subsets from a dataset based on pre-defined split_ids.

    Args:
        dataset (GraphDataset): Dataset with split_ids.

    Returns:
        Tuple[Subset, Subset, Subset]: Train, val, test splits.
    """
    train_indices = [i for i, s in enumerate(dataset.split_ids) if s == 0]
    val_indices = [i for i, s in enumerate(dataset.split_ids) if s == 1]
    test_indices = [i for i, s in enumerate(dataset.split_ids) if s == 2]

    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices),
        torch.utils.data.Subset(dataset, test_indices)
    )
