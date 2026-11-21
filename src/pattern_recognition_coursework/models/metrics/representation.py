"""Frozen Global / DN4 Local scores and fixed, label-free score calibration."""
import torch

from pattern_recognition_coursework.models.metrics.episode import prototype_scores
from pattern_recognition_coursework.models.metrics.local import local_descriptor_scores


def row_score_standardize(scores, eps=1e-6):
    """Population std across candidate classes for each query; never uses labels."""
    if scores.ndim != 2 or scores.shape[1] < 2 or eps <= 0:
        raise ValueError('Expected [queries, classes>=2] and positive epsilon')
    centered = scores-scores.mean(-1, keepdim=True)
    scale = centered.square().mean(-1, keepdim=True).sqrt().clamp_min(eps)
    return centered/scale


def chunked_local_scores(support, query, n_way, k_shot, local_k=3, query_chunk=4):
    if query_chunk < 1 or len(query) < 1:
        raise ValueError('Query count and chunk size must be positive')
    return torch.cat([local_descriptor_scores(support, chunk, n_way, k_shot, local_k)
                      for chunk in query.split(query_chunk)], dim=0)


def representation_scores(global_features, support_local, query_local_cpu, n_way,
                          k_shot, local_k=3, query_chunk=4, eps=1e-6):
    """All arithmetic follows the Global input's device (CUDA in GPU mode).

    The complete local cache remains on CPU. Only support and one query block
    occupy GPU memory. No true query labels enter any scoring/calibration API.
    """
    device = global_features.device
    count = n_way*k_shot
    query_count = len(global_features)-count
    if global_features.dtype != torch.float32 or global_features.shape[1:] != (512,):
        raise ValueError('Global input must be raw FP32 [N,512]')
    if support_local.shape != (count,512,7,7) or query_local_cpu.shape != (query_count,512,7,7):
        raise ValueError('Expected Layer4 [N,512,7,7] in identical episode order')
    if support_local.dtype != torch.float32 or query_local_cpu.dtype != torch.float32:
        raise ValueError('Local input must remain FP32')
    global_scores = prototype_scores(global_features[:count],global_features[count:],n_way,k_shot,'cosine')
    support_device = support_local.to(device)
    local_scores = torch.cat([
        local_descriptor_scores(support_device,chunk.to(device),n_way,k_shot,local_k)
        for chunk in query_local_cpu.split(query_chunk)
    ])
    global_calibrated = row_score_standardize(global_scores,eps)
    local_calibrated = row_score_standardize(local_scores,eps)
    fusion = .5*global_calibrated+.5*local_calibrated
    if not all(x.device==device and x.isfinite().all() for x in
               [global_scores,local_scores,global_calibrated,local_calibrated,fusion]):
        raise ValueError('Score device mismatch or nonfinite scores')
    return dict(global_scores=global_scores,local_scores=local_scores,fusion_scores=fusion,
                global_calibrated=global_calibrated,local_calibrated=local_calibrated)
