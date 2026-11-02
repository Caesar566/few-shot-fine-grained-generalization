"""Classification diagnostics for an episode and equal-weight episode summaries.

Labels refer to the local class slots of each episode, not global species IDs.
"""

import numpy as np
from scipy.special import logsumexp


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def evaluation_metrics(scores, targets, *, from_logits=True, num_bins=15):
    """Return JSON-compatible accuracy, NLL, macro-F1, ECE and confusion counts.

    ``scores`` has shape [queries, classes] and may contain logits or log
    probabilities. Set ``from_logits=False`` for normalized probabilities.
    ECE uses equally spaced confidence bins [lower, upper), with 1 in the last
    bin. Macro-F1 includes every output class, using zero for undefined F1.
    Zero true-class probabilities are clipped to float64's smallest positive
    normal value for a finite, explicitly bounded NLL.
    """
    if isinstance(num_bins, bool) or not isinstance(num_bins, (int, np.integer)) or num_bins < 1:
        raise ValueError("num_bins must be a positive integer")
    values = _numpy(scores).astype(np.float64)
    labels = _numpy(targets)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("scores must be a non-empty [queries, classes] matrix")
    if labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iu":
        raise ValueError("targets must contain one integer class ID per query")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    classes = values.shape[1]
    if ((labels < 0) | (labels >= classes)).any():
        raise ValueError("target class ID is outside the score columns")
    if from_logits:
        log_probabilities = values - logsumexp(values, axis=1, keepdims=True)
        probabilities = np.exp(log_probabilities)
    else:
        if (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1, rtol=1e-5, atol=1e-7):
            raise ValueError("probabilities must be in [0, 1] and each row must sum to 1")
        probabilities = values / values.sum(axis=1, keepdims=True)
        log_probabilities = np.log(np.clip(probabilities, np.finfo(np.float64).tiny, 1))
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * confusion.diagonal(), denominator, out=np.zeros(classes), where=denominator != 0)
    confidence = probabilities.max(axis=1)
    bin_ids = np.minimum((confidence * num_bins).astype(np.int64), num_bins - 1)
    ece = 0.0
    for bin_id in range(num_bins):
        selected = bin_ids == bin_id
        if selected.any():
            ece += float(selected.mean() * abs(correct[selected].mean() - confidence[selected].mean()))
    return {
        "accuracy": float(correct.mean()),
        "nll": float(-log_probabilities[np.arange(len(labels)), labels].mean()),
        "macro_f1": float(f1.mean()),
        "ece": float(ece),
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_labels": "episode-local class slots; rows=true, columns=predicted",
        "samples": len(labels),
        "calibration_bins": int(num_bins),
    }


def aggregate_episode_metrics(episodes):
    """Aggregate diagnostics with one equal vote per episode, preserving scores.

    ECE and macro-F1 are averaged per episode. The summed confusion matrix is
    only a diagnostic for local slots and cannot identify individual species.
    """
    episodes = list(episodes)
    if not episodes:
        raise ValueError("at least one episode is required")
    matrices = [np.asarray(item["confusion_matrix"], dtype=np.int64) for item in episodes]
    if any(matrix.shape != matrices[0].shape for matrix in matrices):
        raise ValueError("episodes must use the same number of class slots")
    if any(item["calibration_bins"] != episodes[0]["calibration_bins"] for item in episodes):
        raise ValueError("episodes must use the same ECE bin count")
    result = {}
    for metric in ("accuracy", "nll", "macro_f1", "ece"):
        values = np.asarray([item[metric] for item in episodes], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite episode {metric}")
        result[metric] = float(values.mean())
    accuracy = [float(item["accuracy"]) for item in episodes]
    result.update({
        "episode_accuracies": accuracy,
        "episode_std": float(np.std(accuracy, ddof=1)) if len(episodes) > 1 else 0.0,
        "episodes": len(episodes),
        "samples": sum(item["samples"] for item in episodes),
        "confusion_matrix": np.stack(matrices).sum(axis=0).tolist(),
        "confusion_matrix_labels": "episode-local class slots; rows=true, columns=predicted",
        "calibration_bins": episodes[0]["calibration_bins"],
        "metric_aggregation": "unweighted mean of episode accuracy, NLL, macro-F1 and ECE",
    })
    return result
