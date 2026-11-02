import pytest
import torch
from torch import nn

from pattern_recognition_coursework.models.few_shot import FewShotModel
from pattern_recognition_coursework.models.metrics.local import local_descriptor_scores
from pattern_recognition_coursework.engine.experiment import configure_trainable_scope


def test_local_matching_uses_support_class_descriptors():
    support = torch.tensor([[[[1., 1.]], [[0., 0.]]],
                            [[[0., 0.]], [[1., 1.]]]])
    query = torch.tensor([[[[1., 1.]], [[0., 0.]]],
                          [[[0., 0.]], [[1., 1.]]]])
    scores = local_descriptor_scores(support, query, n_way=2, k_shot=1, local_k=1)
    assert scores.argmax(dim=1).tolist() == [0, 1]
    with pytest.raises(ValueError, match="local_k"):
        local_descriptor_scores(support, query, 2, 1, local_k=3)


def test_global_local_classifier_and_context_have_finite_gradients():
    local = FewShotModel("global_local", metric="cosine", local_k=1, local_weight=0.5)
    spatial = torch.randn(4, 512, 2, 2, requires_grad=True)
    scores = local.classify_embeddings(spatial, n_way=2, k_shot=1, n_query=1)
    assert scores.shape == (2, 2)
    scores.sum().backward()
    assert torch.isfinite(spatial.grad).all()

    context = FewShotModel("matchingnet_fce", fce_steps=2)
    context.backbone = nn.Identity()
    embedded = torch.randn(4, 512, requires_grad=True)
    log_probabilities = context(embedded, 2, 1, 1)
    assert torch.allclose(log_probabilities.exp().sum(dim=1), torch.ones(2), atol=1e-5)
    log_probabilities[:, 0].sum().backward()
    assert torch.isfinite(embedded.grad).all()


def test_linear_adapter_starts_as_identity_and_can_train():
    model = FewShotModel("protonet", adapter=True)
    model.backbone = nn.Identity()
    embedded = torch.randn(4, 512)
    torch.testing.assert_close(model.encode(embedded), embedded)
    loss = model(embedded, 2, 1, 1).square().sum()
    loss.backward()
    assert model.adapter.weight.grad is not None
    assert torch.isfinite(model.adapter.weight.grad).all()


def test_adapter_scope_trains_fce_context_but_freezes_backbone():
    model = FewShotModel("matchingnet_fce", adapter=True, fce_steps=2)
    names = configure_trainable_scope(model, "adapter")
    assert names
    assert all(not name.startswith("backbone.") for name in names)
    assert any(name.startswith("adapter.") for name in names)
    assert any(name.startswith("context.") for name in names)
