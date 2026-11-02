"""Checks for research comparisons that must keep class ordering and strict weights."""

import pytest
import torch
from torch import nn
from torchvision.models import resnet18

from scripts.analyze_cub_errors import quartile, quartile_boundaries
from scripts.compare_cub_matching import score_episode
from scripts.convert_simclr_resnet18 import extract_encoder


def test_one_shot_matching_attention_has_same_decisions_as_cosine_prototype():
    generator = torch.Generator().manual_seed(17)
    for _ in range(5):
        features = torch.randn(10, 8, 2, 2, generator=generator)
        scores = score_episode(features, 5, 1, 1)
        assert scores["matching_attention"] == scores["prototype_cosine"]


def test_size_quartiles_use_all_images_without_outcome_information():
    boundaries = quartile_boundaries([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert boundaries == [0.3, 0.5, 0.7]
    assert [quartile(value, boundaries) for value in (0.1, 0.3, 0.5, 0.7)] == [1, 2, 3, 4]


def test_simclr_extractor_requires_complete_torchvision_encoder():
    backbone = resnet18(weights=None)
    backbone.fc = nn.Identity()
    inverse = {"conv1": "0", "bn1": "1", "layer1": "4", "layer2": "5", "layer3": "6", "layer4": "7"}
    source = {"backbone." + inverse[key.split(".", 1)[0]] + "." + key.split(".", 1)[1]: value
              for key, value in backbone.state_dict().items()}
    assert extract_encoder({"state_dict": source}).keys() == backbone.state_dict().keys()
    source.pop("backbone.0.weight")
    with pytest.raises(ValueError, match="不完整"):
        extract_encoder({"state_dict": source})
