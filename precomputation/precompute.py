# Parts of the code inspired by https://trimesh.org/shortest.html (MIT License)
# All licenses available in the license folder

import multiprocessing
import os
import sys

import pyfqmr
import trimesh
import numpy as np
from scipy.spatial._ckdtree import cKDTree
from collections import deque

import shutil
import networkx as nx
import heapq
from tqdm import tqdm


def get_nearest_points(m1, m2):
    """
    Computes the closest points on mesh `m1` for each vertex in mesh `m2`.

    Args:
        m1 (trimesh.Trimesh): Reference mesh.
        m2 (trimesh.Trimesh): Mesh whose vertices are used to find nearest points on `m1`.

    Returns:
        ndarray: Array of closest points on `m1` corresponding to vertices of `m2`.
    """
    nearest_points, _, _ = m1.nearest.on_surface(m2.vertices)
    return nearest_points

def iterative_fast_simplification(in_mesh, target_reduction, num_iterations=1, output_mesh_prefix=None):
    """
    Simplifies a mesh over multiple iterations and records nearest-point mappings after each decimation step.

    Args:
        in_mesh (trimesh.Trimesh): Original mesh to simplify.
        target_reduction (float): Proportional reduction in number of faces.
        num_iterations (int): Number of simplification steps.
        output_mesh_prefix (str or None): If provided, saves each intermediate mesh to disk with this prefix.

    Returns:
        Tuple:
            - List[Trimesh]: Decimated meshes after each iteration.
            - List[ndarray]: Nearest point mappings from current to previous mesh.
    """
    mesh = in_mesh
    decimated_meshes = []
    nearest_points_array = []
    for i in range(num_iterations):
        mesh_simplifier = pyfqmr.Simplify()
        mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
        mesh_simplifier.simplify_mesh(target_count=int(len(mesh.faces) * target_reduction), aggressiveness=7, preserve_border=True, verbose=True)
        vertices, faces, normals = mesh_simplifier.getMesh()
        raw = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh2 = trimesh.util.concatenate([m for m in raw.split(only_watertight=False) if len(m.vertices) > 50])
        if output_mesh_prefix is not None:
            output_mesh_path =  f"{output_mesh_prefix}_iter{i + 1}.obj"
            mesh2.export(output_mesh_path)
        nearest_points = get_nearest_points(mesh, mesh2)
        decimated_meshes.append(mesh2)
        nearest_points_array.append(nearest_points)
        mesh = mesh2
    return decimated_meshes, nearest_points_array


def find_top_k_closest_neighbors(g: nx.Graph, start_node: int, k: int = 10):
    """
    Finds the k closest nodes in a graph based on edge-weighted geodesic distance.
    Implementation of Dijkstra's algorithm with early stopping on k+1 nodes 

    Args:
        g (networkx.Graph): Graph where edge weights represent distances.
        start_node (int): Node from which distances are computed.
        k (int): Number of neighbors to retrieve.

    Returns:
        Tuple:
            - nodes (ndarray): IDs of the closest nodes.
            - distances (ndarray): Adjusted distances (geodesic - first hop).
            - first_hops (ndarray): First hop neighbor used in the shortest path to each node.
    """
    pq = [(0, start_node, start_node)]
    dist = {}
    first_hop={}

    while pq:
        d, node, first = heapq.heappop(pq)
        if node in dist:
            continue
        dist[node] = d
        first_hop[node] = first
        if len(dist) == k + 1:
            break
        for neighbor in g.neighbors(node):
            if neighbor not in dist:
                weight = g[node][neighbor].get('length', 1)
                heapq.heappush(pq, (d + weight, neighbor, first if node != start_node else neighbor))

    nodes_list = list(dist.keys())
    distances = list(dist.values())
    padding_length = k - len(nodes_list) + 1
    nodes = np.array(nodes_list + [start_node] * padding_length)
    distances = np.array(distances + [np.nan] * padding_length)
    first_hop_list = [first_hop[n] for n in nodes_list] + [start_node] * padding_length
    first_hop_distances = np.array([dist[f] for f in first_hop_list])
    return nodes[:k + 1], distances[:k + 1] - first_hop_distances, np.array(first_hop_list)[:k + 1]
    
    
def sort_matrix_pairs(nodes, distances):
    """
    Sorts node-distance pairs along the row dimension by ascending distance.

    Args:
        nodes (ndarray): 2D array of node indices.
        distances (ndarray): 2D array of distances.

    Returns:
        Tuple[ndarray, ndarray]: Sorted nodes and distances.
    """
    sorting_indices = np.argsort(distances, axis=1)
    sorted_distances = np.take_along_axis(distances, sorting_indices, axis=1)
    sorted_nodes = np.take_along_axis(nodes, sorting_indices, axis=1)

    return sorted_nodes, sorted_distances
    
def compute_distances(mesh, points):
    """
    Computes geodesic distances on the mesh surface between input points and their k nearest neighbors.

    Args:
        mesh (trimesh.Trimesh): Mesh used for graph construction and distance computation.
        points (ndarray): Array of input 3D points.

    Returns:
        Tuple:
            - distances (ndarray): Geodesic distances to top-k neighbors.
            - indices (ndarray): Indices of top-k neighbors.
            - nearest_vertices (List[int]): Closest mesh vertex to each input point.
    """
    kd_tree = cKDTree(mesh.vertices)
    k_points = 10
    g = nx.Graph()
    edges = mesh.edges_unique
    length = mesh.edges_unique_length
    indices_geodesic = []
    distances_geodesic = []
    nearest_vertices = []
    for edge, L in zip(edges, length):
        g.add_edge(*edge, length=L)

    for i, point in enumerate(points):
        nearest_vertex_idx = kd_tree.query(point)[1]
        nearest_vertices.append(nearest_vertex_idx)
        topk_geodesic, dist_geodesic, first_hops = find_top_k_closest_neighbors(g, nearest_vertex_idx, k=k_points-1)
        first_hop_correction = np.linalg.norm(mesh.vertices[first_hops] - np.array([point]*k_points), axis=1)
        dist_geodesic += first_hop_correction
        distances_geodesic.append(dist_geodesic)
        indices_geodesic.append(topk_geodesic)

    sorted_nodes, sorted_distances = sort_matrix_pairs(np.array(indices_geodesic), np.array(distances_geodesic))
    return sorted_distances, sorted_nodes, nearest_vertices
    
    
def normalize_distance_matrix_forward(distances):
    """
    Normalizes distance matrix into forward weights using reciprocal and sum normalization.

    Args:
        distances (ndarray): Distance matrix where the first column corresponds to the closest nodes.

    Returns:
        ndarray: Normalized weights.
    """
    mask = distances[:, 0] == 0.0
    rcpr = np.reciprocal(distances[mask, 1:])
    ntn = np.nan_to_num(rcpr, nan=0.0)
    ss = np.sum(ntn, axis=1)
    distances[mask, 0] = 1.0 / (19 * ss)
    reciprocal = np.zeros_like(distances)
    reciprocal = np.reciprocal(distances, where=~np.isnan(distances), out=reciprocal)   
    weights = reciprocal / np.sum(reciprocal, axis=1)[:, np.newaxis]
    return weights
    
    
def normalize_distance_matrix_backward(distances):
    """
    Normalizes distance matrix into backwards weights using reciprocal and sum normalization.

    Args:
        distances (ndarray):  Distance matrix where the first column corresponds to the closest nodes.

    Returns:
        ndarray: Normalized reciprocal weights.
    """
    reciprocal = np.zeros_like(distances)
    reciprocal = np.reciprocal(distances, where=~np.isnan(distances), out=reciprocal)
    reciprocal[np.isnan(reciprocal)] = 0.0
    row_sums = np.sum(reciprocal, axis=1, keepdims=True)
    zero_rows = row_sums == 0
    weights = np.zeros_like(reciprocal)
    weights = np.divide(reciprocal, row_sums, out=weights, where=~zero_rows)
    return weights

def compute_unpooling_weights_sparse(neighbor_indices, distances, b, global_limit):
    """
    Computes unpooling weights by inverting the neighbor mapping.

    Args:
        neighbor_indices (ndarray): Indices of neighbors in lower-resolution graph.
        distances (ndarray): Corresponding geodesic distances.
        b (int): Number of high-resolution vertices.
        global_limit (int): Number of reverse neighbors to retain.

    Returns:
        Tuple:
            - weights (ndarray): Normalized unpooling weights.
            - indices (ndarray): Reverse index mapping for unpooling.
    """
    z = int(np.max(np.unique(neighbor_indices, return_counts=True)[1]))
    z = max(z, global_limit)
    index_matrix = [deque([0] * z, maxlen=z) for _ in range(b)]
    distance_matrix = [deque([np.nan] * z, maxlen=z)  for _ in range(b)]
    for i, row in enumerate(neighbor_indices):
        for j, val in enumerate(row):
            index_matrix[val].appendleft(i)
            distance_matrix[val].appendleft(distances[i][j])
    npDist = np.array(distance_matrix)
    npIndex = np.array(index_matrix)
    sorted_indices = np.argsort(npDist, axis=1)
    sorted_distance_matrix = np.take_along_axis(npDist, sorted_indices, axis=1)[:, :global_limit]
    sorted_indices_matrix = np.take_along_axis(npIndex, sorted_indices, axis=1)[:, :global_limit]
    return normalize_distance_matrix_backward(sorted_distance_matrix), sorted_indices_matrix


def precompute_weights(input_mesh_path, folder_path, iterations, limit, decimation_ratio=0.7):
    """
    Computes and stores structural pooling parameters for a mesh by progressive simplification.

    Args:
        input_mesh_path (str): Path to the input .stl mesh file.
        folder_path (str): Output directory to store weights and indices.
        iterations (int): Number of simplification stages.
        limit (int): Number of neighbors to keep in pooling/unpooling matrices.
        decimation_ratio (float): Ratio of faces to retain per simplification step (default=0.7).

    Saves:
        - `edge_index_*.npy`
        - `weights_*.npy`
        - `reverse_weights_*.npy` in `folder_path`
    """
    input_mesh = trimesh.load_mesh(input_mesh_path)
    decimated_meshes, nearest_points_array = iterative_fast_simplification(input_mesh, decimation_ratio, iterations,
                                                                           output_mesh_prefix=f"{folder_path}/")
    decimated_meshes.insert(0, input_mesh)
    for i in range(iterations):
        m1 = decimated_meshes[i]
        edge_index = decimated_meshes[i+1].edges_unique
        nearest_points = nearest_points_array[i]
        distances_geodesic, indices_geodesic, _ = compute_distances(m1, nearest_points)
        weights = normalize_distance_matrix_forward(distances_geodesic[:, :limit])
        reverse_weight_matrix, reverse_indices_matrix  = compute_unpooling_weights_sparse(indices_geodesic, distances_geodesic, len(m1.vertices), limit)
        if reverse_weight_matrix.shape[1] != limit:
            print(f"input_mesh_path: {input_mesh_path}, {i}")
        np.save(f"{folder_path}/edge_index_{i}.npy", np.array(edge_index))
        np.save(f"{folder_path}/weights_{i}.npy", np.array([indices_geodesic[:, :limit], weights])) # Caution: converts integer indices to floats
        np.save(f"{folder_path}/reverse_weights_{i}.npy", np.array([reverse_indices_matrix, reverse_weight_matrix])) # Caution: converts integer indices to floats

def process_file(args):
    """
    Worker function for multiprocessing that processes a single .stl file and computes its structural parameters.

    Args:
        args (Tuple[str, str, int]): Tuple containing (file name, directory path, number of iterations).

    Returns:
        str: Name of the processed file (for progress tracking).
    """
    file, organ_path, iters = args
    if not file.endswith(".stl"):
        print(f"not a .stl file: {file}")
        return
    folder = organ_path+"/"+file.split("/")[-1].split(".stl")[0]
    if os.path.exists(folder):
        print(f"folder {folder} already exists!")
        if folder == "in":
            shutil.rmtree(folder)
        else:
            return
    os.makedirs(folder)
    try:
        precompute_weights(os.path.join(organ_path, file), folder, iters, 5)
    except Exception as e:
        print(folder)
        print(e)
        raise
    return file

if __name__ == "__main__":
    """
    Batch processing pipeline to compute structural pooling parameters for all .stl files in a directory.

    Usage:
        python script.py <organ_path> <iterations>

    Parallelizes computation across available CPU cores using multiprocessing.
    """
    np.seterr(divide='ignore', invalid='ignore')
    if len(sys.argv) < 3:
        print("Usage: python script.py <organ_path> <iterations>")
        exit(1)
    organ_path = sys.argv[1]
    iters = int(sys.argv[2])
    stl_files = [f for f in os.listdir(organ_path) if f.endswith(".stl")]
    multi = True
    if multi:
        num_workers = min(len(stl_files), os.cpu_count())
        with multiprocessing.Pool(num_workers) as pool:
            with tqdm(total=len(stl_files), desc="Processing STL files", unit="file") as pbar:
                for _ in pool.imap_unordered(process_file, [(file, organ_path, iters) for file in stl_files]):
                    pbar.update(1)



