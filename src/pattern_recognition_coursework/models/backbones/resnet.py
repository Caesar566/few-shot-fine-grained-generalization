"""ResNet features and explicit, strictly checked local weight import."""

import hashlib
import warnings
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def load_backbone_weights(
    backbone: nn.Module, path: str | Path, prefix: str = ""
) -> dict:
    """Load a trusted local tensor state_dict without guessing encoder prefixes.

    A raw dictionary, or one under exactly one of ``state_dict``/``model``, is
    accepted. An explicit prefix selects an encoder from a larger model. Every
    encoder tensor must match exactly. Only missing counters registered on real
    BatchNorm modules are initialized to zero for legacy torchvision weights;
    a complete, well-shaped classifier pair may be discarded. Both are logged.
    """
    if not isinstance(prefix, str) or (prefix and not prefix.endswith(".")):
        raise ValueError("backbone_prefix 必须为空或以点结束，例如 encoder. 或 backbone.")
    path = Path(path).resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("本地权重必须为 state_dict 或包含 state_dict/model 的字典")
    containers = [key for key in ("state_dict", "model") if isinstance(payload.get(key), Mapping)]
    if len(containers) > 1:
        raise ValueError("权重同时包含 state_dict 与 model；请导出唯一明确的 state_dict")
    container = containers[0] if containers else None
    state = payload[container] if container else payload
    if not all(isinstance(key, str) for key in state):
        raise ValueError("state_dict 的键必须为字符串")
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    outside = sorted(key for key in state if not key.startswith(prefix))
    if not selected or not all(isinstance(value, torch.Tensor) for value in selected.values()):
        raise ValueError("指定前缀未选中完整的张量 state_dict")

    ignored_head = []
    head_keys = {"fc.weight", "fc.bias"} & selected.keys()
    if head_keys:
        if head_keys != {"fc.weight", "fc.bias"}:
            raise ValueError("分类头必须同时包含 fc.weight 与 fc.bias")
        weight, bias = selected["fc.weight"], selected["fc.bias"]
        if weight.ndim != 2 or weight.shape[1] != 512 or bias.shape != (weight.shape[0],):
            raise ValueError("分类头形状不是 ResNet-18 的完整线性分类头")
        ignored_head = [prefix + key for key in sorted(head_keys)]
        for key in head_keys:
            del selected[key]

    expected = backbone.state_dict()
    bn_counters = {
        f"{name}.num_batches_tracked" if name else "num_batches_tracked"
        for name, module in backbone.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and module.num_batches_tracked is not None
    }
    initialized_buffers = sorted((expected.keys() - selected.keys()) & bn_counters)
    missing = sorted(expected.keys() - selected.keys() - set(initialized_buffers))
    unexpected = sorted(selected.keys() - expected.keys())
    mismatched = [
        key for key in expected.keys() & selected.keys()
        if expected[key].shape != selected[key].shape
    ]
    dtype_mismatched = [
        key for key in expected.keys() & selected.keys()
        if expected[key].dtype != selected[key].dtype
    ]
    if missing or unexpected or mismatched or dtype_mismatched:
        raise ValueError(
            f"backbone 权重不完整或不兼容: missing={missing}, "
            f"unexpected={unexpected}, shape_mismatch={sorted(mismatched)}, "
            f"dtype_mismatch={sorted(dtype_mismatched)}"
        )
    for key in initialized_buffers:
        selected[key] = torch.zeros_like(expected[key])
    backbone.load_state_dict(selected, strict=True)
    if initialized_buffers:
        warnings.warn(
            "兼容历史 BatchNorm checkpoint，将以下缺失 num_batches_tracked 缓冲区初始化为 0: "
            + ", ".join(initialized_buffers),
            UserWarning,
            stacklevel=2,
        )
    ignored = outside + ignored_head
    if ignored:
        warnings.warn(
            f"显式加载 backbone，排除 {len(outside)} 个前缀外张量和 "
            f"{len(ignored_head)} 个分类头张量；完整名单见 backbone_load_info['ignored_keys']",
            UserWarning,
            stacklevel=2,
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path), "sha256": digest.hexdigest(), "container": container,
        "prefix": prefix, "loaded_keys": len(selected) - len(initialized_buffers),
        "initialized_buffers": initialized_buffers, "ignored_keys": ignored,
    }


def make_backbone(
    pretrained: bool = False,
    backbone_weights: str | Path | None = None,
    backbone_prefix: str = "",
) -> nn.Module:
    """Return a 512-dimensional ResNet-18 while preserving torchvision key names."""
    if pretrained and backbone_weights is not None:
        raise ValueError("pretrained 与 backbone_weights 不可同时指定")
    if backbone_prefix and backbone_weights is None:
        raise ValueError("backbone_prefix 需要同时指定本地 backbone_weights")
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()
    if backbone_weights is not None:
        backbone.backbone_load_info = load_backbone_weights(
            backbone, backbone_weights, backbone_prefix)
    elif pretrained:
        cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
        backbone.backbone_load_info = {
            "source": "torchvision", "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "url": weights.url, "sha256": hashlib.sha256(cached.read_bytes()).hexdigest(),
        }
    else:
        backbone.backbone_load_info = None
    return backbone


def spatial_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run ResNet up to layer4, before average pooling and the classifier."""
    features = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(images))))
    for layer in (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4):
        features = layer(features)
    return features


class SupervisedResNet18(nn.Module):
    """Existing 512-d encoder with a retained CUB base-class training head."""

    def __init__(self, num_classes: int, backbone_weights=None):
        super().__init__()
        if num_classes < 2:
            raise ValueError("监督分类至少需要两个类别")
        self.backbone = make_backbone(backbone_weights=backbone_weights)
        self.classifier = nn.Linear(512, num_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(images))
