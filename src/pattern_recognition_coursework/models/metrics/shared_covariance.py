"""Train-only pooled within-class diagonal covariance; no query inputs or inverse."""
import math

import torch


def pooled_diagonal_covariance(train_features, train_labels, shrinkage=.1,
                               relative_jitter=1e-4, absolute_jitter=1e-8):
    """Unbiased within-class variances, shrunk towards an isotropic diagonal.

    Fit once using labeled CUB TRAIN features in eval mode, never query labels.
    S_d = sum_c sum_i (x_ci[d]-mean_c[d])^2 / (N-C).
    v_d = (1-alpha)*S_d + alpha*mean(S) + max(abs_eps, rel_eps*mean(S)).
    No feature centering/normalizing is applied to Support or Query by this fit.
    """
    if train_features.ndim != 2 or train_features.shape[0] != len(train_labels):
        raise ValueError('Expected train features [N,D] and labels [N]')
    if train_labels.ndim != 1 or train_labels.dtype != torch.int64:
        raise ValueError('Training labels must be int64 [N]')
    if not 0 <= shrinkage <= 1 or relative_jitter < 0 or absolute_jitter <= 0:
        raise ValueError('Invalid covariance regularization')
    if not all(math.isfinite(v) for v in [shrinkage,relative_jitter,absolute_jitter]):
        raise ValueError('Covariance regularization must be finite')
    x=train_features.detach().cpu().double();labels=train_labels.detach().cpu()
    if not x.isfinite().all():raise ValueError('Nonfinite train features')
    classes=torch.unique(labels);df=len(x)-len(classes)
    if df <= 0:raise ValueError('Within-class covariance requires positive N-C')
    residual_sum=torch.zeros(x.shape[1],dtype=torch.float64)
    counts={}
    for label in classes:
        members=x[labels==label];counts[str(int(label))]=len(members)
        residual_sum+=(members-members.mean(0)).square().sum(0)
    raw=residual_sum/df
    scale=float(raw.mean());jitter=max(absolute_jitter,relative_jitter*scale)
    diagonal=(1-shrinkage)*raw+shrinkage*scale+jitter
    if not diagonal.isfinite().all() or not (diagonal>0).all():
        raise ValueError('Covariance diagonal must be finite and strictly positive')
    metadata=dict(estimator='pooled_within_class_shrunk_diagonal',fit_split='CUB Train',
                  sample_count=len(x),classes=len(classes),degrees_of_freedom=df,
                  class_counts=counts,shrinkage=shrinkage,relative_jitter=relative_jitter,
                  absolute_jitter=absolute_jitter,actual_jitter=jitter,
                  raw_variance_mean=scale,raw_variance_min=float(raw.min()),raw_variance_max=float(raw.max()),
                  diagonal_min=float(diagonal.min()),diagonal_max=float(diagonal.max()),
                  condition_number_diagonal=float(diagonal.max()/diagonal.min()),
                  used_query_features=False,used_query_labels=False,used_nabirds=False,
                  arithmetic='float64 estimation; elementwise division, no matrix inverse',
                  formula='v=(1-alpha)*pooled_within_class_variance+alpha*mean_variance+jitter')
    return diagonal,raw,metadata


def shared_diagonal_mahalanobis_scores(support,query,n_way,k_shot,diagonal):
    """Raw support means, then squared diagonal Mahalanobis distance / D.

    Prototype arithmetic remains float32 when input features are float32. Only
    the metric's subtraction/division accumulation is promoted to float64.
    """
    if support.ndim!=2 or query.ndim!=2 or support.shape[1]!=query.shape[1]:
        raise ValueError('Support/query must have the same feature dimension')
    if support.shape[0]!=n_way*k_shot or diagonal.shape!=(support.shape[1],):
        raise ValueError('Support count or covariance shape mismatch')
    if not diagonal.isfinite().all() or not (diagonal>0).all():
        raise ValueError('Covariance diagonal must be finite and strictly positive')
    prototypes=support.reshape(n_way,k_shot,-1).mean(1)
    difference=query.double()[:,None,:]-prototypes.double()[None,:,:]
    distance=(difference.square()/diagonal.double().to(query.device)).sum(-1)
    scores=(-distance/support.shape[-1]).to(support.dtype)
    if not scores.isfinite().all():raise ValueError('Nonfinite Mahalanobis scores')
    return scores
