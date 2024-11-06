import glob
import os.path as osp

import torch
from scipy.optimize import linear_sum_assignment
# warnings.warn("Unable to load pointnet2_ops cpp extension. JIT Compiling.")


def match(source, target, dist_th=0.005, hungarian=False):
    """
    Hungarian matching between source and target points
    Args:
        source: [B, N, 3] or [N, 3] - Batch of source point clouds or a single point cloud
        target: List[] - Batch of target point clouds or a single point cloud
    Returns:
        distance: [B, N] or [N] - Minimum distance from each source point to the closest target point
        index: [B, N] or [N] - Index of the closest target point for each source point
    """
    if not isinstance(source, list):
        if source.dim() < 3:
            source = [source.detach()]  # Add batch dimension, resulting in [1, N, 3]
        else:
            source = [
                src.detach() for src in source
            ]  # Add batch dimension, resulting in [B, N, 3]

    if not isinstance(target, list):
        if target.dim() < 3:
            target = [target]  # Add batch dimension, resulting in [1, N, 3]
        else:
            target = [
                tg for tg in target
            ]  # Add batch dimension, resulting in [B, N, 3]

    batch_size = source.__len__()
    device = source[0].device

    indices = []

    for bt in range(batch_size):
        # Calculate the pairwise distances between source and target points
        pairwise_distance = torch.cdist(target[bt], source[bt]).detach()  # [N, M]

        # Use the Hungarian algorithm to find the optimal matching
        if hungarian:
            row_ind, col_ind = linear_sum_assignment(pairwise_distance.cpu().numpy())

            # Convert indices to tensors
            col_ind = torch.tensor(col_ind, device=device)
            row_ind = torch.tensor(row_ind, device=device)

            # Gather the corresponding distances
            dist = pairwise_distance[row_ind, col_ind]
            inside_ball = dist < dist_th
            ind = torch.stack([col_ind[inside_ball], row_ind[inside_ball]])
        else:
            # find the distance inside the ball, all targets inside the ball will be assigned to the same source
            pairwise_distance[pairwise_distance > dist_th] = float("inf")
            min_dist, min_ind = torch.min(pairwise_distance, dim=-1)
            valid = min_dist < float("inf")
            col_ind = min_ind[valid]
            row_ind = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            ind = torch.stack((col_ind, row_ind), dim=1).permute(1, 0)

        # Append results
        indices.append(ind)

    return indices


def gram_schmidt(vectors):
    """
    Perform the Gram-Schmidt process on a set of vectors.

    Args:
        vectors (torch.Tensor): A tensor of shape (n, d) where n is the number of vectors
                                and d is the dimension of each vector.

    Returns:
        torch.Tensor: A tensor of orthogonal vectors with the same shape as the input.
    """
    n, d = vectors.shape
    ortho_vectors = torch.zeros_like(vectors)

    for i in range(n):
        # Start with the current vector
        ortho_vectors[i] = vectors[i]

        # Subtract the projection of the current vector onto the previous orthogonal vectors
        for j in range(i):
            proj = torch.dot(ortho_vectors[j], vectors[i]) / torch.dot(
                ortho_vectors[j], ortho_vectors[j]
            )
            ortho_vectors[i] -= proj * ortho_vectors[j]

        # Normalize the vector to ensure it has unit length (if needed)
        ortho_vectors[i] = ortho_vectors[i] / torch.norm(ortho_vectors[i])

    return ortho_vectors

