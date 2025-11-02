"""
Metrics computation for firewall evaluation.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats


def compute_trigger_rate(similarities: np.ndarray, is_leaky: np.ndarray, threshold: float) -> float:
    """
    Compute trigger rate: fraction of outputs blocked at threshold.

    Args:
        similarities: Array of max similarities per query
        is_leaky: Boolean array of leakage labels (unused, for compatibility)
        threshold: Detection threshold

    Returns:
        Fraction of queries that would be blocked
    """
    return np.mean(similarities >= threshold)


def compute_leakage_rate(is_leaky: np.ndarray) -> float:
    """
    Compute leakage rate: fraction of outputs that are leaky.

    Args:
        is_leaky: Boolean array of leakage labels

    Returns:
        Fraction of leaky outputs
    """
    return np.mean(is_leaky)


def compute_blocked_leakage(similarities: np.ndarray, is_leaky: np.ndarray,
                           threshold: float) -> float:
    """
    Compute blocked-leakage rate: fraction of leaky outputs caught by firewall.
    This is the recall on leaks.

    Args:
        similarities: Array of max similarities per query
        is_leaky: Boolean array of leakage labels
        threshold: Detection threshold

    Returns:
        Fraction of leaky outputs that are blocked (recall)
    """
    leaky_indices = is_leaky.astype(bool)
    if not leaky_indices.any():
        return 0.0

    leaky_sims = similarities[leaky_indices]
    return np.mean(leaky_sims >= threshold)


def compute_false_positive_rate(similarities: np.ndarray, is_leaky: np.ndarray,
                                threshold: float) -> float:
    """
    Compute false positive rate: fraction of non-leaky outputs blocked.

    Args:
        similarities: Array of max similarities per query
        is_leaky: Boolean array of leakage labels
        threshold: Detection threshold

    Returns:
        Fraction of non-leaky outputs that are blocked
    """
    non_leaky_indices = ~is_leaky.astype(bool)
    if not non_leaky_indices.any():
        return 0.0

    non_leaky_sims = similarities[non_leaky_indices]
    return np.mean(non_leaky_sims >= threshold)


def bootstrap_metric(similarities: np.ndarray, is_leaky: np.ndarray,
                     threshold: float, metric_fn, n_bootstrap: int = 1000,
                     random_state: int = 42) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        similarities: Array of max similarities per query
        is_leaky: Boolean array of leakage labels
        threshold: Detection threshold
        metric_fn: Function to compute metric (takes similarities, is_leaky, threshold)
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        (mean, lower_95ci, upper_95ci)
    """
    rng = np.random.RandomState(random_state)
    n = len(similarities)
    bootstrap_values = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = rng.choice(n, size=n, replace=True)
        boot_sims = similarities[indices]
        boot_leaky = is_leaky[indices]

        # Compute metric
        value = metric_fn(boot_sims, boot_leaky, threshold)
        bootstrap_values.append(value)

    bootstrap_values = np.array(bootstrap_values)
    mean = np.mean(bootstrap_values)
    lower = np.percentile(bootstrap_values, 2.5)
    upper = np.percentile(bootstrap_values, 97.5)

    return mean, lower, upper


def compute_all_metrics(similarities: np.ndarray, is_leaky: np.ndarray,
                       threshold: float, n_bootstrap: int = 1000) -> Dict:
    """
    Compute all metrics at a given threshold with bootstrap CIs.

    Args:
        similarities: Array of max similarities per query
        is_leaky: Boolean array of leakage labels
        threshold: Detection threshold
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary of metrics with confidence intervals
    """
    metrics = {}

    # Point estimates
    tr = compute_trigger_rate(similarities, is_leaky, threshold)
    lr = compute_leakage_rate(is_leaky)
    bl = compute_blocked_leakage(similarities, is_leaky, threshold)
    fpr = compute_false_positive_rate(similarities, is_leaky, threshold)

    # Bootstrap CIs
    tr_mean, tr_lower, tr_upper = bootstrap_metric(
        similarities, is_leaky, threshold, compute_trigger_rate, n_bootstrap
    )
    bl_mean, bl_lower, bl_upper = bootstrap_metric(
        similarities, is_leaky, threshold, compute_blocked_leakage, n_bootstrap
    )
    fpr_mean, fpr_lower, fpr_upper = bootstrap_metric(
        similarities, is_leaky, threshold, compute_false_positive_rate, n_bootstrap
    )

    metrics = {
        'threshold': threshold,
        'trigger_rate': tr,
        'trigger_rate_ci': (tr_lower, tr_upper),
        'leakage_rate': lr,
        'blocked_leakage': bl,
        'blocked_leakage_ci': (bl_lower, bl_upper),
        'false_positive_rate': fpr,
        'false_positive_rate_ci': (fpr_lower, fpr_upper),
    }

    return metrics


def mcnemar_test(detector1_blocked: np.ndarray, detector2_blocked: np.ndarray,
                 is_leaky: np.ndarray) -> Dict:
    """
    Perform McNemar's test comparing two detectors.

    Args:
        detector1_blocked: Boolean array of what detector 1 blocked
        detector2_blocked: Boolean array of what detector 2 blocked
        is_leaky: Boolean array of leakage labels

    Returns:
        Dictionary with test statistics
    """
    # Focus on leaky samples only
    leaky_mask = is_leaky.astype(bool)

    d1_leaky = detector1_blocked[leaky_mask]
    d2_leaky = detector2_blocked[leaky_mask]

    # Contingency table
    # b: detector 1 caught, detector 2 missed
    # c: detector 1 missed, detector 2 caught
    b = np.sum(d1_leaky & ~d2_leaky)
    c = np.sum(~d1_leaky & d2_leaky)

    # McNemar's test
    if b + c == 0:
        return {'statistic': 0.0, 'p_value': 1.0, 'b': b, 'c': c}

    statistic = (abs(b - c) - 1) ** 2 / (b + c)  # With continuity correction
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        'statistic': statistic,
        'p_value': p_value,
        'b': int(b),
        'c': int(c)
    }
