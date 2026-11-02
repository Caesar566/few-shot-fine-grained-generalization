"""Episode classifiers sharing one support/query convention."""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from pattern_recognition_coursework.models.backbones.resnet import make_backbone, spatial_features
from pattern_recognition_coursework.models.metrics.context import FullContextEmbedding
from pattern_recognition_coursework.models.metrics.local import local_descriptor_scores

from pattern_recognition_coursework.models.metrics.episode import (
    prototype_scores,
    standardize_from_support,
)


class FewShotModel(nn.Module):
    def __init__(
        self,
        method: str = "protonet",
        pretrained: bool = False,
        metric: str = "euclidean",
        episode_standardization: bool = False,
        shrinkage: float = 0.5,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        backbone_weights: str | Path | None = None,
        backbone_prefix: str = "",
        local_k: int = 3,
        local_weight: float = 0.5,
        fce_steps: int = 5,
        adapter: bool = False,
    ):
        super().__init__()
        if method not in {"protonet", "matchingnet", "matchingnet_fce", "dn4", "global_local"}:
            raise ValueError(f"未知方法: {method}")
        if metric not in {"euclidean", "cosine", "mahalanobis"}:
            raise ValueError(f"未知距离: {metric}")
        if method in {"matchingnet", "matchingnet_fce", "dn4"} and metric != "euclidean":
            raise ValueError("匹配基线固定使用余弦相似度；距离选项仅适用于原型及融合的全局分支")
        if not 0 < shrinkage <= 1:
            raise ValueError("协方差收缩系数必须在 (0, 1] 内")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("温度必须大于 0")
        if isinstance(local_k, bool) or not isinstance(local_k, int) or local_k < 1:
            raise ValueError("local_k 必须是正整数")
        if not 0 <= local_weight <= 1:
            raise ValueError("local_weight 必须在 [0, 1] 内")
        if isinstance(fce_steps, bool) or not isinstance(fce_steps, int) or fce_steps < 1:
            raise ValueError("fce_steps 必须是正整数")
        if method == "dn4" and episode_standardization:
            raise ValueError("dn4 使用局部描述子余弦匹配，不支持全局 episode 标准化")
        self.method = method
        self.metric = metric
        self.episode_standardization = episode_standardization
        self.shrinkage = shrinkage
        self.local_k = local_k
        self.local_weight = local_weight
        self.backbone = make_backbone(pretrained, backbone_weights, backbone_prefix)
        self.backbone_load_info = self.backbone.backbone_load_info
        self.context = FullContextEmbedding(512, fce_steps) if method == "matchingnet_fce" else None
        self.adapter = nn.Linear(512, 512, bias=False) if adapter else None
        if self.adapter is not None:
            nn.init.eye_(self.adapter.weight)
        self.model_config = {
            "method": method, "pretrained": pretrained, "metric": metric,
            "episode_standardization": episode_standardization, "shrinkage": shrinkage,
            "temperature": temperature, "learnable_temperature": learnable_temperature,
            "backbone_weights": str(backbone_weights) if backbone_weights is not None else None,
            "backbone_prefix": backbone_prefix, "local_k": local_k,
            "local_weight": local_weight, "fce_steps": fce_steps,
            "adapter": adapter,
        }
        log_temperature = torch.tensor(math.log(temperature))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(log_temperature)
        else:
            self.register_buffer("log_temperature", log_temperature)

    def forward(
        self, images: torch.Tensor, n_way: int, k_shot: int, n_query: int
    ) -> torch.Tensor:
        """Images are support grouped by class, then query grouped by class."""
        self._validate_episode(images, n_way, k_shot, n_query)
        return self.classify_embeddings(self.encode(images), n_way, k_shot, n_query)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return image features; cache only in eval mode with fixed transforms.

        Local methods return [N,512,H,W]; other methods return [N,512]. The FCE
        transformation remains episode-specific and is deliberately not cached.
        """
        if self.method in {"dn4", "global_local"}:
            features = spatial_features(self.backbone, images)
            if self.adapter is not None:
                features = self.adapter(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return features
        features = self.backbone(images)
        return self.adapter(features) if self.adapter is not None else features

    @staticmethod
    def _validate_episode(values: torch.Tensor, n_way: int, k_shot: int, n_query: int) -> None:
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1
               for x in (n_way, k_shot, n_query)):
            raise ValueError("way/shot/query 必须是正整数")
        if values.ndim < 2 or values.shape[0] != n_way * (k_shot + n_query):
            raise ValueError("episode 图像数与 way/shot/query 配置不一致")

    def classify_embeddings(
        self, embeddings: torch.Tensor, n_way: int, k_shot: int, n_query: int
    ) -> torch.Tensor:
        """Classify precomputed per-image features in the same episode order."""
        self._validate_episode(embeddings, n_way, k_shot, n_query)
        support_count = n_way * k_shot
        query_count = n_way * n_query
        support = embeddings[:support_count]
        query = embeddings[support_count:]
        temperature = self.log_temperature.clamp(math.log(1e-3), math.log(1e3)).exp()
        local_scores = None
        if self.method in {"dn4", "global_local"}:
            local_scores = local_descriptor_scores(support, query, n_way, k_shot, self.local_k)
            if self.method == "dn4":
                return local_scores / temperature
            support, query = support.mean(dim=(-1, -2)), query.mean(dim=(-1, -2))
        elif embeddings.ndim != 2:
            raise ValueError("全局方法需要二维图像特征")
        if self.episode_standardization:
            support, query = standardize_from_support(support, query)
        if self.method in {"protonet", "global_local"}:
            scores = prototype_scores(
                support, query, n_way, k_shot, self.metric, self.shrinkage
            )
            if local_scores is not None:
                scores = (1 - self.local_weight) * scores + self.local_weight * local_scores
            return scores / temperature

        if self.context is not None:
            support, query = self.context(support, query)
        # Matching Networks: cosine attention over individual support images.
        support = F.normalize(support, dim=-1)
        query = F.normalize(query, dim=-1)
        attention = ((query @ support.T) / temperature).softmax(dim=-1)
        labels = torch.arange(n_way, device=embeddings.device).repeat_interleave(k_shot)
        probabilities = attention.new_zeros(query_count, n_way)
        probabilities.scatter_add_(1, labels.expand(query_count, -1), attention)
        return probabilities.clamp_min(1e-12).log()
