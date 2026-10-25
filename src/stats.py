"""Inference: clustered bootstrap, multiplicity adjustment, design effects.

Resampling is at the level of image groups, not images, so that
near-duplicate frames cannot be split across a bootstrap replicate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Clustered bootstrap
# --------------------------------------------------------------------------

def clustered_bootstrap(df: pd.DataFrame, statistic, group_col: str = "group_id",
                        n_boot: int = 2000, seed: int = 20260101,
                        alpha: float = 0.05):
    """Percentile bootstrap CI, resampling whole groups with replacement.

    `statistic` takes a DataFrame and returns a float. Returns
    (point estimate, lo, hi, array of replicate values).
    """
    rng = np.random.default_rng(seed)
    groups = df[group_col].to_numpy()
    uniq = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in uniq}

    point = float(statistic(df))
    reps = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_by_group[g] for g in drawn])
        reps[b] = statistic(df.iloc[idx])

    lo, hi = np.percentile(reps[~np.isnan(reps)],
                           [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), reps


def bootstrap_p(reps: np.ndarray, null_value: float = 0.0) -> float:
    """Two-sided P by inverting the bootstrap distribution.

    Cannot fall below 2/(B+1); a P reported at that floor means no
    replicate crossed the null, not a precisely estimated small value.
    """
    reps = reps[~np.isnan(reps)]
    b = len(reps)
    frac = min((reps <= null_value).mean(), (reps >= null_value).mean())
    return float(max(2 * frac, 2.0 / (b + 1)))


def cluster_permutation_p(values_a: np.ndarray, values_b: np.ndarray,
                          n_perm: int = 2000, seed: int = 20260101) -> float:
    """Exact-style permutation test for a difference in means.

    Labels are permuted at the observation level within the pooled set;
    use only where the grouping is cohort-level.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:n_a].mean() - pooled[n_a:].mean()) >= abs(observed):
            count += 1
    return float((count + 1) / (n_perm + 1))


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------

def holm_bonferroni(pvalues) -> list[float]:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()


def benjamini_hochberg(pvalues) -> list[float]:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, m / (rank + 1) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()


# --------------------------------------------------------------------------
# Design-effect sensitivity analysis
# --------------------------------------------------------------------------

def design_effect(m: float, icc: float) -> float:
    """Kish design effect for a fixed cluster size m."""
    return 1.0 + (m - 1.0) * icc


def rescale_interval(estimate: float, lo: float, hi: float,
                     m: float, icc: float):
    """Widen a CI by sqrt(DEFF), keeping any asymmetry about the estimate."""
    factor = np.sqrt(design_effect(m, icc))
    return (estimate + (lo - estimate) * factor,
            estimate + (hi - estimate) * factor)


def images_per_patient(n_images: int, n_patients: int) -> float:
    """Cluster size for a cohort.

    Use the documented per-cohort patient counts. Dividing the 2D image
    count by the whole-dataset patient count understates clustering for
    the 2D cohort and overstates it for CEUS.
    """
    if not n_patients:
        raise ValueError("Patient count unavailable for this cohort.")
    return n_images / n_patients


# --------------------------------------------------------------------------
# Selective prediction
# --------------------------------------------------------------------------

def risk_coverage_curve(uncertainty: np.ndarray, risk: np.ndarray):
    """Risk among retained images as coverage falls from 1 to 1/n."""
    order = np.argsort(np.asarray(uncertainty, float), kind="mergesort")
    r = np.asarray(risk, float)[order]
    k = np.arange(1, len(r) + 1)
    return k / len(r), np.cumsum(r) / k


def aurc(uncertainty: np.ndarray, risk: np.ndarray):
    """(AURC, AURC expected under random referral)."""
    _, curve = risk_coverage_curve(uncertainty, risk)
    return float(curve.mean()), float(np.asarray(risk, float).mean())


def operating_point(uncertainty: np.ndarray, failure: np.ndarray,
                    threshold: float) -> dict:
    """Confusion matrix and derived rates at a referral threshold."""
    u = np.asarray(uncertainty, float)
    f = np.asarray(failure, bool)
    referred = u >= threshold
    tp = int((referred & f).sum())
    fp = int((referred & ~f).sum())
    fn = int((~referred & f).sum())     # silent failures
    tn = int((~referred & ~f).sum())
    n = tp + fp + fn + tn
    safe = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return {
        "n": n, "flagged_failures": tp, "false_positive_referrals": fp,
        "silent_failures": fn, "correctly_retained": tn,
        "sensitivity": safe(tp, tp + fn), "specificity": safe(tn, tn + fp),
        "ppv": safe(tp, tp + fp), "npv": safe(tn, tn + fn),
        "referral_rate": safe(tp + fp, n),
        "failure_rate_all": safe(tp + fn, n),
        "failure_rate_retained": safe(fn, fn + tn),
    }


def youden_threshold(uncertainty: np.ndarray, failure: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1.

    Fit on validation data ONLY. Re-running this on test or CEUS data
    invalidates the prespecification.
    """
    u = np.asarray(uncertainty, float)
    f = np.asarray(failure, bool)
    best, best_j = float("nan"), -np.inf
    for t in np.unique(u):
        op = operating_point(u, f, t)
        j = op["sensitivity"] + op["specificity"] - 1
        if j > best_j:
            best, best_j = float(t), j
    return best
