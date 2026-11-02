"""DN4-style cosine nearest-neighbor matching from images to support classes."""

import torch
import torch.nn.functional as F


def local_descriptor_scores(
    support: torch.Tensor,
    query: torch.Tensor,
    n_way: int,
    k_shot: int,
    local_k: int = 3,
) -> torch.Tensor:
    """Return mean top-k cosine similarity, pooling support patches per class.

    Inputs are [images, channels, height, width]. Each query patch retrieves its
    k nearest descriptors from all support images of a candidate class. Averaging
    instead of summing the HW*k terms makes the scale stable across resolutions.
    """
    if support.ndim != 4 or query.ndim != 4 or support.shape[1] != query.shape[1]:
        raise ValueError("局部特征必须为通道数一致的四维张量")
    if n_way < 1 or k_shot < 1 or support.shape[0] != n_way * k_shot:
        raise ValueError("support 数量与 way/shot 不一致")
    if min(*support.shape[1:], *query.shape[1:]) < 1:
        raise ValueError("局部特征的通道和空间尺寸必须大于 0")
    count = k_shot * support.shape[2] * support.shape[3]
    if isinstance(local_k, bool) or not isinstance(local_k, int) or not 1 <= local_k <= count:
        raise ValueError(f"local_k 必须在 1 与每类 support 描述子数 {count} 之间")
    support = F.normalize(support.flatten(2).transpose(1, 2), dim=-1)
    query = F.normalize(query.flatten(2).transpose(1, 2), dim=-1)
    classes = support.reshape(n_way, count, support.shape[-1])
    # One class at a time bounds the largest similarity tensor to Q*HW*(K*HW).
    return torch.stack([
        (query @ descriptors.T).topk(local_k, dim=-1).values.mean(dim=(-1, -2))
        for descriptors in classes
    ], dim=-1)
