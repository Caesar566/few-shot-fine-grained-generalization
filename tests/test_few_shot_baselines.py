import pytest
import torch
from torch import nn

from pattern_recognition_coursework.data.cub_dataset import FewShotBatchSampler
from pattern_recognition_coursework.models.few_shot import FewShotModel


class DummyDataset:
    class_to_indices = {0: list(range(10)), 1: list(range(10, 20)), 2: list(range(20, 30))}


def test_episode_sampling_is_reproducible_and_disjoint():
    options = dict(dataset=DummyDataset(), episodes_per_epoch=4, n_way=2, k_shot=2, n_query=3, seed=12)
    sampler = FewShotBatchSampler(**options)
    episodes = list(sampler)
    assert episodes == list(sampler)
    assert episodes != list(FewShotBatchSampler(**{**options, "seed": 13}))
    for episode in episodes:
        assert len(episode) == len(set(episode)) == 10
        support, query = episode[:4], episode[4:]
        assert not set(support) & set(query)
        support_classes = [index // 10 for index in support[::2]]
        assert [index // 10 for index in query[::3]] == support_classes


@pytest.mark.parametrize("method", ["protonet", "matchingnet"])
def test_baseline_logits_follow_episode_class_order(method):
    model = FewShotModel(method)
    model.backbone = nn.Identity()
    embeddings = torch.tensor([
        [1.0, 0.0], [1.1, 0.0],  # class 0 support
        [0.0, 1.0], [0.0, 1.1],  # class 1 support
        [1.0, 0.1], [0.1, 1.0],  # class 0 and 1 queries
    ])
    scores = model(embeddings, n_way=2, k_shot=2, n_query=1)
    assert scores.shape == (2, 2)
    assert scores.argmax(dim=1).tolist() == [0, 1]
    assert torch.isfinite(scores).all()


def test_sampler_rejects_impossible_episode():
    with pytest.raises(ValueError, match="需要 4 类"):
        FewShotBatchSampler(DummyDataset(), 1, 4, 1, 1)
