import torch

from pattern_recognition_coursework.models.metrics.episode import (
    prototype_scores,
    standardize_from_support,
)


def test_support_only_standardization_does_not_depend_on_other_queries():
    support = torch.tensor([[1.0, 2.0], [3.0, 2.0]])
    first_query = torch.tensor([[2.0, 3.0]])
    normalized_support, normalized_query = standardize_from_support(support, first_query)
    changed_support, changed_queries = standardize_from_support(
        support, torch.cat([first_query, torch.tensor([[1000.0, -1000.0]])])
    )
    torch.testing.assert_close(normalized_support, changed_support)
    torch.testing.assert_close(normalized_query[0], changed_queries[0])
    assert torch.isfinite(normalized_support).all()  # constant second feature


def test_mahalanobis_low_rank_matches_direct_covariance_inverse():
    support = torch.tensor([[1.0, 0.0, 0.2], [1.2, 0.1, 0.0],
                            [0.0, 1.0, 0.1], [0.1, 1.1, 0.0]], dtype=torch.float64)
    query = torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1]], dtype=torch.float64)
    shrinkage = 0.4
    actual = prototype_scores(support, query, 2, 2, "mahalanobis", shrinkage)
    centered = support - support.mean(dim=0)
    covariance = centered.T @ centered / (len(support) - 1)
    scale = (covariance.trace() / support.shape[1]).clamp_min(1e-4)
    covariance = (1 - shrinkage) * covariance + (shrinkage * scale + 1e-5) * torch.eye(3)
    prototypes = support.reshape(2, 2, 3).mean(dim=1)
    delta = query[:, None, :] - prototypes[None, :, :]
    expected = -torch.einsum("qnd,df,qnf->qn", delta, covariance.inverse(), delta) / 3
    torch.testing.assert_close(actual, expected)


def test_singular_support_covariance_has_finite_gradients():
    support = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], requires_grad=True)
    query = torch.tensor([[0.9, 0.1]], requires_grad=True)
    scores = prototype_scores(support, query, 2, 2, "mahalanobis", 0.5)
    scores.sum().backward()
    assert torch.isfinite(scores).all()
    assert torch.isfinite(support.grad).all()
    assert torch.isfinite(query.grad).all()
