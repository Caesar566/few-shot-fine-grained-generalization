"""Confidence interval across sampled episodes, conditional on one fixed model."""
import numpy as np
from scipy.stats import t


def accuracy_summary(accuracies):
    values = np.asarray(accuracies, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('At least two finite episode accuracies are required')
    if ((values < 0) | (values > 1)).any():
        raise ValueError('Episode accuracy must be in [0,1]')
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    sem = std/np.sqrt(n)
    critical = float(t.ppf(.975, n-1))
    half = float(critical*sem)
    return dict(episodes=n, mean_accuracy=mean, sample_std=std, standard_error=float(sem),
                ci95=[mean-half, mean+half], ci95_half_width=half,
                ci_method='two-sided Student-t, df=n-1, sample std ddof=1',
                ci_unit='episode; conditional on fixed checkpoint, CUB Val and sampling protocol',
                t_critical=critical, minimum=float(values.min()), maximum=float(values.max()))


def paired_accuracy_difference(first, second):
    """Second minus first on identically ordered episodes; CI uses paired deltas."""
    accuracy_summary(first);accuracy_summary(second)
    a=np.asarray(first,dtype=np.float64);b=np.asarray(second,dtype=np.float64)
    if a.shape!=b.shape:raise ValueError('Paired accuracy arrays must have identical shape')
    delta=b-a;n=len(delta);mean=float(delta.mean());std=float(delta.std(ddof=1))
    half=float(t.ppf(.975,n-1)*std/np.sqrt(n))
    return dict(episodes=n,mean_delta=mean,sample_std_delta=std,ci95=[mean-half,mean+half],
                ci95_half_width=half,direction='second minus first',
                unit='episode accuracy difference; report in percentage points',
                ci_method='two-sided Student-t on paired per-episode deltas, ddof=1')
