import csv
import os
import numpy as np
import torch
import trimesh
from torch_geometric.data import Dataset, Data
import torch_geometric
from sklearn.model_selection import train_test_split


class GraphDatasetStructural(Dataset):
    """
    Custom PyTorch Geometric Dataset to handle loading mesh and precomputed structural pooing data from a folder-based class layout.

    Each class is represented by a subdirectory, and each sample is a .stl file with associated precomputed
    structural pooling parameters stored as .npy files.

    Args:
        root (str): Root directory containing subfolders for each class label.
        transform (callable, optional): Optional transform to apply on a Data object.
        pre_transform (callable, optional): Optional pre-transform applied before saving to disk.
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDatasetStructural, self).__init__(root, transform, pre_transform)
        self.graph_paths = []
        self.labels = []
        self.struct_params_folders = []
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
                class_dir = os.path.dirname(full_path)
                class_name = os.path.basename(class_dir)
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = next_idx
                    next_idx += 1
                label_idx = class_to_idx[class_name]
                self.struct_params_folders.append(mesh_path.split(".stl")[0])
                self.graph_paths.append(mesh_path)
                self.labels.append(label_idx)
                self.split_ids.append(split_idx)
        self.label_dict = {v: k for k, v in class_to_idx.items()}

    def len(self):
        return len(self.graph_paths)

    def get(self, idx):
        features, edge_index, faces = extract_vertex_features_and_edges(self.graph_paths[idx])
        weights, reverse_weights, edge_indices, n = extract_struct_params(self.struct_params_folders[idx])
        forward1= torch.from_numpy(np.concatenate([weights[0][0], weights[0][1]]))
        forward2 = torch.from_numpy(np.concatenate([weights[1][0], weights[1][1]]))
        forward3 = torch.from_numpy(np.concatenate([weights[2][0], weights[2][1]]))
        backward1 = torch.from_numpy(np.concatenate([reverse_weights[0][0], reverse_weights[0][1]]))
        backward2 = torch.from_numpy(np.concatenate([reverse_weights[1][0], reverse_weights[1][1]]))
        backward3 = torch.from_numpy(np.concatenate([reverse_weights[2][0], reverse_weights[2][1]]))
        label = self.labels[idx]
        edge_indices1 = torch_geometric.utils.to_undirected(torch.from_numpy(edge_indices[0]).T).T
        edge_indices2 = torch_geometric.utils.to_undirected(torch.from_numpy(edge_indices[1]).T).T
        edge_indices3 = torch_geometric.utils.to_undirected(torch.from_numpy(edge_indices[2]).T).T
        data = Data(x=torch.from_numpy(features.copy()),
                    edge_index=torch_geometric.utils.to_undirected(torch.from_numpy(edge_index.copy())),
                    y=torch.tensor([label], dtype=torch.long),
                    faces=faces,
                    path=self.graph_paths[idx],
                    forward1=forward1,
                    stage1_lengths=len(weights[0][0]),
                    forward2=forward2,
                    stage2_lengths=len(weights[1][0]),
                    forward3=forward3,
                    stage3_lengths=len(weights[2][0]),
                    backward1=backward1,
                    backward2=backward2,
                    backward3=backward3,
                    edge_indices1 = edge_indices1,
                    edge_indices2 = edge_indices2,
                    edge_indices3 = edge_indices3,
                    edge_indices_length1 = torch.tensor([len(edge_indices1)]),
                    edge_indices_length2 = torch.tensor([len(edge_indices2)]),
                    edge_indices_length3 = torch.tensor([len(edge_indices3)]))
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
    edge_index = mesh.edges_unique.T
    return np.hstack([scale_and_shift_points(mesh.vertices, mesh.edges_unique_length.mean()), mesh.vertex_normals], dtype=np.float32), edge_index, mesh.faces

def extract_struct_params(folder_path):
    """
    Loads precomputed structural pooling parameters from a given folder.

    Args:
        folder_path (str): Path to the folder containing .npy files for weights, reverse weights, and edge indices.

    Returns:
        Tuple:
            - weights (List[Tuple[ndarray, ndarray]]): Forward structural pooling index-weight pairs for each stage.
            - reverse_weights (List[Tuple[ndarray, ndarray]]): Reverse structural pooling index-weight pairs for each stage.
            - edge_indices (List[ndarray]): Edge indices for each pooling level.
            - n (int): Number of stages detected based on file naming.
    """
    files = os.listdir(folder_path)
    weights, reverse_weights, edge_indices = [], [], []
    n = max(int(f.split('_')[-1].split('.')[0]) for f in files if f.startswith("weights_"))
    for i in range(n + 1):
        weights.append(np.load(os.path.join(folder_path, f"weights_{i}.npy")))
        reverse_weights.append(np.load(os.path.join(folder_path, f"reverse_weights_{i}.npy")))
        edge_indices.append(np.load(os.path.join(folder_path, f"edge_index_{i}.npy")))

    return weights, reverse_weights, edge_indices, n

def split_dataset(dataset,):
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
