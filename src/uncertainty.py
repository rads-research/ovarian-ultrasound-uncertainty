"""Uncertainty measures.

`ensemble_disagreement` is the prespecified primary measure. The others
were compared with it in an exploratory analysis added after the locked
results were examined.

Throughout, `probs` has shape (n_members, H, W) and holds per-member
foreground probabilities for one image.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from metrics import dice

EPS = 1e-12


def _binarise(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.asarray(probs) >= threshold


def ensemble_disagreement(probs: np.ndarray, threshold: float = 0.5) -> float:
    """PRESPECIFIED PRIMARY MEASURE.

    One minus the mean pairwise Dice between binarised member predictions.
    0 means every member produced an identical contour; values approach 1
    as members disagree. Requires at least two members.
    """
    masks = _binarise(probs, threshold)
    n = len(masks)
    if n < 2:
        raise ValueError("Ensemble disagreement needs at least two members.")
    pairwise = [dice(masks[i], masks[j]) for i, j in combinations(range(n), 2)]
    return float(1.0 - np.mean(pairwise))


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def mean_predictive_entropy(probs: np.ndarray, threshold: float = 0.5,
                            dilation_iters: int = 10) -> float:
    """PRESPECIFIED SECONDARY MEASURE.

    Mean binary entropy of the ensemble-mean probability map, evaluated
    over the union of the binarised member masks dilated by
    `dilation_iters` iterations -- not over the whole frame. A whole-frame
    mean is dominated by confidently empty background and is close to zero
    for every image, which destroys the discrimination the measure is
    being judged on.
    """
    probs = np.asarray(probs)
    roi = lesion_roi(probs, threshold, dilation_iters)
    return float(_binary_entropy(probs.mean(axis=0))[roi].mean())


def pixel_std(probs: np.ndarray) -> float:
    """Exploratory: mean across pixels of the per-pixel member SD."""
    return float(np.asarray(probs).std(axis=0).mean())


def mutual_information(probs: np.ndarray) -> float:
    """Exploratory: total predictive entropy minus mean member entropy."""
    probs = np.asarray(probs)
    total = _binary_entropy(probs.mean(axis=0))
    aleatoric = _binary_entropy(probs).mean(axis=0)
    return float((total - aleatoric).mean())


def area_variability(probs: np.ndarray, threshold: float = 0.5) -> float:
    """Exploratory: coefficient of variation of predicted lesion area."""
    areas = _binarise(probs, threshold).reshape(len(probs), -1).sum(axis=1)
    mean = areas.mean()
    if mean == 0:
        return 0.0
    return float(areas.std() / mean)


def single_model_entropy(prob: np.ndarray) -> float:
    """Exploratory, one member: mean per-pixel predictive entropy."""
    return float(_binary_entropy(np.asarray(prob)).mean())


def single_model_one_minus_max(prob: np.ndarray) -> float:
    """Exploratory, one member: 1 - mean maximum class probability."""
    prob = np.asarray(prob)
    return float(1.0 - np.maximum(prob, 1 - prob).mean())


def all_measures(probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Every measure for one image.

    Single-model values are averaged across all members, so that no member
    is selected; the manuscript reports them this way throughout.
    """
    probs = np.asarray(probs)
    return {
        "disagreement": ensemble_disagreement(probs, threshold),
        "mean_entropy": mean_predictive_entropy(probs),
        "pixel_std": pixel_std(probs),
        "mutual_information": mutual_information(probs),
        "area_variability": area_variability(probs, threshold),
        "single_model_entropy": float(np.mean(
            [single_model_entropy(p) for p in probs])),
        "single_model_one_minus_max": float(np.mean(
            [single_model_one_minus_max(p) for p in probs])),
    }


def lesion_roi(probs: np.ndarray, threshold: float = 0.5,
               dilation_iters: int = 10) -> np.ndarray:
    """Lesion region used for region-restricted calibration.

    Union of binarised member predictions, dilated by `dilation_iters`
    iterations of a 3x3 square element. If no member predicts any
    foreground, the whole image is returned, so the region is never empty.
    """
    from scipy import ndimage

    union = _binarise(probs, threshold).any(axis=0)
    if not union.any():
        return np.ones_like(union, dtype=bool)
    return ndimage.binary_dilation(
        union, structure=np.ones((3, 3), bool), iterations=dilation_iters)


def expected_calibration_error(prob: np.ndarray, truth: np.ndarray,
                               n_bins: int = 15):
    """ECE with equal-width bins, plus the per-bin reliability table."""
    prob = np.asarray(prob, float).ravel()
    truth = np.asarray(truth, bool).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_bins - 1)

    ece, rows = 0.0, []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            rows.append({"bin_index": b, "mean_pred": np.nan,
                         "observed_frequency": np.nan, "n_pixels": 0})
            continue
        mean_pred = float(prob[sel].mean())
        obs = float(truth[sel].mean())
        ece += (n / len(prob)) * abs(obs - mean_pred)
        rows.append({"bin_index": b, "mean_pred": mean_pred,
                     "observed_frequency": obs, "n_pixels": n})
    return float(ece), rows


def brier_score(prob: np.ndarray, truth: np.ndarray) -> float:
    prob = np.asarray(prob, float).ravel()
    truth = np.asarray(truth, bool).ravel().astype(float)
    return float(np.mean((prob - truth) ** 2))
