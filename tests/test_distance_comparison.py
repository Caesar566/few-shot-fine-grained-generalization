"""Metric-only invariants, within-class covariance, stability and paired statistics."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.stats import t

from pattern_recognition_coursework.analysis.episodic_statistics import paired_accuracy_difference
from pattern_recognition_coursework.engine.distance_comparison import FixedMetricScorer
from pattern_recognition_coursework.models.metrics.episode import prototype_scores
from pattern_recognition_coursework.models.metrics.shared_covariance import (
    pooled_diagonal_covariance,shared_diagonal_mahalanobis_scores,
)


def test_cosine_means_raw_support_before_norm_division():
    support=torch.zeros(4,512);support[:,:2]=torch.tensor([[100.,0.],[0.,1.],[0.,2.],[0.,4.]])
    query=torch.zeros(2,512);query[:,:2]=torch.tensor([[1.,1.],[1.,2.]])
    before=support.clone();features=torch.cat([support,query])
    actual=FixedMetricScorer('cosine').classify_embeddings(features,2,2,1)
    p=support.reshape(2,2,512).mean(1)
    expected=(query@p.T)/(query.norm(dim=1)[:,None]*p.norm(dim=1)[None,:])
    torch.testing.assert_close(actual,expected)
    wrong_p=F.normalize(support,dim=1).reshape(2,2,512).mean(1)
    wrong=F.normalize(query,dim=1)@F.normalize(wrong_p,dim=1).T
    assert actual[1].argmax()!=wrong[1].argmax()
    assert torch.equal(support,before)
    assert torch.equal(actual,prototype_scores(support,query,2,2,'cosine'))


def test_covariance_is_within_class_and_uses_n_minus_c():
    x=torch.tensor([[-1.,0.],[1.,0.],[99.,10.],[101.,10.]])
    v,raw,m=pooled_diagonal_covariance(x,torch.tensor([0,0,1,1]),.25,0,.01)
    torch.testing.assert_close(raw,torch.tensor([2.,0.],dtype=torch.float64),rtol=0,atol=0)
    torch.testing.assert_close(v,torch.tensor([1.76,.26],dtype=torch.float64))
    assert m['degrees_of_freedom']==2 and m['used_query_labels'] is False
    # Large between-class offsets have NO effect on the within-class covariance.
    shifted=x.clone();shifted[2:]+=5000
    v2,_,_=pooled_diagonal_covariance(shifted,torch.tensor([0,0,1,1]),.25,0,.01)
    torch.testing.assert_close(v,v2,rtol=0,atol=0)


def test_mahalanobis_matches_hand_diagonal_distance_with_raw_means():
    support=torch.tensor([[0.,0.],[2.,0.],[0.,2.],[2.,2.]])
    query=torch.tensor([[1.,1.],[3.,2.]])
    v=torch.tensor([4.,1.],dtype=torch.float64)
    actual=shared_diagonal_mahalanobis_scores(support,query,2,2,v)
    expected=torch.tensor([[-.5,-.5],[-2.5,-.5]])
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_singular_and_zero_variance_are_stable_without_inverse(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Matrix inverse must never be called')
    monkeypatch.setattr(torch.linalg,'inv',forbidden)
    monkeypatch.setattr(torch,'inverse',forbidden)
    x=torch.zeros(4,512)
    diagonal,_,m=pooled_diagonal_covariance(x,torch.tensor([0,0,1,1]))
    assert (diagonal>0).all() and m['condition_number_diagonal']==1
    scores=shared_diagonal_mahalanobis_scores(x,torch.ones(2,512),2,2,diagonal)
    assert scores.isfinite().all()


@pytest.mark.parametrize('options',[dict(shrinkage=-.1),dict(shrinkage=1.1),dict(absolute_jitter=0),dict(relative_jitter=-1)])
def test_invalid_regularization_rejected(options):
    with pytest.raises(ValueError):
        pooled_diagonal_covariance(torch.zeros(4,2),torch.tensor([0,0,1,1]),**options)


def test_paired_difference_uses_delta_variance_and_correct_sign():
    a=[.2,.4,.8];b=[.4,.4,.6];d=np.array(b)-np.array(a)
    r=paired_accuracy_difference(a,b)
    expected=float(t.ppf(.975,2)*d.std(ddof=1)/np.sqrt(3))
    np.testing.assert_allclose(r['mean_delta'],d.mean(),atol=1e-15)
    np.testing.assert_allclose(r['ci95'],[d.mean()-expected,d.mean()+expected])
    assert paired_accuracy_difference(a,a)['ci95']==[0.,0.]
    with pytest.raises(ValueError):paired_accuracy_difference(a,b[:2])


def test_euclidean_cannot_be_silently_rescored():
    with pytest.raises(ValueError,match='reused'):FixedMetricScorer('euclidean')
