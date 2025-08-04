import torch
def split_matrices_by_lengths(matrix, lengths):
    """
    Splits a concatenated tensor of alternating index and weight values into separate index and weight matrices.

    Args:
        matrix (Tensor): Concatenated sequence of index-weight pairs (1D tensor).
        lengths (Tensor): Number of index-weight pairs in each group (1D tensor of ints).

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing two tensors:
            - index_matrix (LongTensor): Extracted index values.
            - weight_matrix (Tensor): Extracted weight values.
    """
    indices = torch.tensor([0,1], device=lengths.device).repeat(len(lengths))
    indices = indices.repeat_interleave(lengths.repeat_interleave(2))
    odd_mask = indices % 2 == 1
    even_mask = indices % 2 == 0

    return matrix[even_mask].long(), matrix[odd_mask]

def offset_from_lengths(lengths):
    """
    Computes starting offsets for segments based on their lengths.

    Args:
        lengths (Tensor): 1D tensor containing the lengths of each segment.

    Returns:
        Tensor: A tensor of starting offsets for each segment.
    """
    return torch.concatenate((torch.tensor([0], dtype=lengths.dtype).to(lengths.device), torch.cumsum(lengths, 0)))[:-1]

def process_global_matrices(batch):
    """
    Processes a batch object containing structural pooling matrices and constructs corrected index/weight tensors.

    Extracts forward and backward precomputed matrices, applies offsets to align node indices across batches,
    and returns corrected matrices, corresponding weights, batch assignments, and pooled edge indices.

    Args:
        batch: A data object with attributes including forward/backward matrices, lengths, and edge index metadata.

    Returns:
        Tuple:
            - List[Tensor]: Corrected index matrices for forward and backward stages.
            - List[Tensor]: Corresponding weight matrices.
            - List[Tensor]: Batch assignments for each stage (used for attentional aggregation).
            - List[Tensor]: Corrected edge indices for each pooling level.
    """
    stage0_length = torch.bincount(batch.batch)
    index1, weight1 = split_matrices_by_lengths(batch.forward1, batch.stage1_lengths)
    index2, weight2 = split_matrices_by_lengths(batch.forward2, batch.stage2_lengths)
    index3, weight3 = split_matrices_by_lengths(batch.forward3, batch.stage3_lengths)
    r_index3, r_weight3 = split_matrices_by_lengths(batch.backward3, batch.stage2_lengths)
    r_index2, r_weight2 = split_matrices_by_lengths(batch.backward2, batch.stage1_lengths)
    r_index1, r_weight1 = split_matrices_by_lengths(batch.backward1, stage0_length)
    stage1_offsets = offset_from_lengths(batch.stage1_lengths)
    stage2_offsets = offset_from_lengths(batch.stage2_lengths)
    stage3_offsets = offset_from_lengths(batch.stage3_lengths)
    corrected_edge_indices = [correct_feature(batch.edge_indices1, stage1_offsets, batch.edge_indices_length1)[0].T,
                              correct_feature(batch.edge_indices2, stage2_offsets, batch.edge_indices_length2)[0].T,
                              correct_feature(batch.edge_indices3, stage3_offsets, batch.edge_indices_length3)[0].T]
    all_node_offsets = [batch.ptr[:-1], stage1_offsets, stage2_offsets, stage3_offsets, stage2_offsets, stage1_offsets]
    all_sizes=[batch.stage1_lengths, batch.stage2_lengths, batch.stage3_lengths, batch.stage2_lengths, batch.stage1_lengths, stage0_length]
    corrected_matrices = []
    batch_ts = []
    for global_matrix, sizes, node_offsets in zip([index1, index2, index3, r_index3, r_index2, r_index1], all_sizes, all_node_offsets):
        corrected_matrix, graph_assignment = correct_feature(global_matrix, node_offsets, sizes)
        corrected_matrices.append(corrected_matrix)
        batch_ts.append(graph_assignment)
    return corrected_matrices, [weight1, weight2, weight3, r_weight3, r_weight2, r_weight1], batch_ts[:3], corrected_edge_indices


def correct_feature(feature_tensor, stage_offsets, sizes):
    """
    Adjusts feature indices to account for batch-wise offsets.

    Args:
        feature_tensor (Tensor): Tensor of indices to correct.
        stage_offsets (Tensor): Offset tensor indicating where each graph starts in the batched representation.
        sizes (Tensor): Number of entries per graph.

    Returns:
        Tuple[Tensor, Tensor]:
            - corrected_matrix (Tensor): Offset-corrected version of `feature_tensor`.
            - graph_assignment (Tensor): Tensor assigning each entry to a corresponding graph in the batch.
    """
    graph_assignment = torch.arange(len(stage_offsets), device=stage_offsets.device).repeat_interleave(sizes)
    corrected_matrix = feature_tensor.clone()
    corrected_matrix += stage_offsets[graph_assignment].view(-1, 1)
    return corrected_matrix, graph_assignment
