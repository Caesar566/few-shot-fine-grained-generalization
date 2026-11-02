"""Distances computed from one episode without using query labels or statistics."""

import math

import torch
import torch.nn.functional as F


def standardize_from_support(
    support: torch.Tensor, query: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply support-only mean and per-feature scale to both sets."""
    mean = support.mean(dim=0, keepdim=True)
    scale = support.var(dim=0, unbiased=False, keepdim=True).add(eps).sqrt()
    return (support - mean) / scale, (query - mean) / scale


def prototype_scores(
    support: torch.Tensor,
    query: torch.Tensor,
    n_way: int,
    k_shot: int,
    metric: str = "euclidean",
    shrinkage: float = 0.5,
) -> torch.Tensor:
    """Return higher-is-better scores for class prototypes."""
    if metric not in {"euclidean", "cosine", "mahalanobis"}:
        raise ValueError(f"未知距离: {metric}")
    if support.ndim != 2 or query.ndim != 2 or support.shape[1] != query.shape[1]:
        raise ValueError("support/query 必须是特征维度相同的二维张量")
    if support.shape[0] != n_way * k_shot:
        raise ValueError("support 数量与 way/shot 不一致")
    prototypes = support.reshape(n_way, k_shot, -1).mean(dim=1)
    if metric == "cosine":
        return F.normalize(query, dim=-1) @ F.normalize(prototypes, dim=-1).T

    differences = query[:, None, :] - prototypes[None, :, :]
    if metric == "euclidean":
        return -differences.square().sum(dim=-1) / support.shape[-1]

    if not 0 < shrinkage <= 1:
        raise ValueError("协方差收缩系数必须在 (0, 1] 内")
    centered = support - support.mean(dim=0, keepdim=True)
    n_support, dimension = support.shape
    denominator = max(n_support - 1, 1)
    trace_scale = (centered.square().sum() / (denominator * dimension)).clamp_min(1e-4)
    diagonal = shrinkage * trace_scale + 1e-5
    low_rank = centered.T * math.sqrt((1 - shrinkage) / denominator)
    kernel = torch.eye(n_support, dtype=support.dtype, device=support.device)
    kernel = kernel + (low_rank.T @ low_rank) / diagonal
    projection = differences @ low_rank
    solved = torch.linalg.solve(kernel, projection.reshape(-1, n_support).T).T
    squared = differences.square().sum(dim=-1) / diagonal
    correction = (projection.reshape(-1, n_support) * solved).sum(dim=-1)
    squared = squared - correction.reshape_as(squared) / diagonal.square()
    return -squared.clamp_min(0) / dimension
