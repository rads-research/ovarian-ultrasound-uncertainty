"""Segmentation metrics, defined exactly as reported in the manuscript.

Masks are boolean arrays of identical shape. Distances are in pixels,
because physical pixel spacing is not published with the dataset.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    """Dice similarity coefficient. Two empty masks score 1.0."""
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    denom = pred.sum() + ref.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, ref).sum() / denom)


def iou(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    union = np.logical_or(pred, ref).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, ref).sum() / union)


def _surface(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels: foreground pixels with a background 4-neighbour."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros_like(mask)
    eroded = ndimage.binary_erosion(mask, structure=np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool), border_value=0)
    return mask & ~eroded


def _surface_distances(pred: np.ndarray, ref: np.ndarray):
    sp, sr = _surface(pred), _surface(ref)
    if not sp.any() or not sr.any():
        return None, None
    # Distance from every pixel to the nearest surface pixel of the other mask
    dt_to_ref = ndimage.distance_transform_edt(~sr)
    dt_to_pred = ndimage.distance_transform_edt(~sp)
    return dt_to_ref[sp], dt_to_pred[sr]


def hd95(pred: np.ndarray, ref: np.ndarray) -> float:
    """95th percentile symmetric Hausdorff distance, in pixels."""
    d_pr, d_rp = _surface_distances(pred, ref)
    if d_pr is None:
        return float("nan")
    return float(max(np.percentile(d_pr, 95), np.percentile(d_rp, 95)))


def nsd(pred: np.ndarray, ref: np.ndarray, tolerance: float = 3.0) -> float:
    """Normalised surface distance at a pixel tolerance."""
    d_pr, d_rp = _surface_distances(pred, ref)
    if d_pr is None:
        return float("nan")
    return float(((d_pr <= tolerance).sum() + (d_rp <= tolerance).sum())
                 / (len(d_pr) + len(d_rp)))


def abs_relative_area_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """|predicted area - reference area| / reference area."""
    a_ref = np.asarray(ref, bool).sum()
    if a_ref == 0:
        return float("nan")
    return float(abs(np.asarray(pred, bool).sum() - a_ref) / a_ref)


def equivalent_diameter_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """Area-derived relative equivalent-diameter error.

    |sqrt(A_pred / A_ref) - 1|. This assumes approximately isotropic
    scaling of lesion dimensions; it is not a measured maximal diameter.
    Exploratory: added after the locked results were examined.
    """
    a_ref = np.asarray(ref, bool).sum()
    if a_ref == 0:
        return float("nan")
    return float(abs(np.sqrt(np.asarray(pred, bool).sum() / a_ref) - 1.0))


def solidity(mask: np.ndarray) -> float:
    """Mask area divided by convex-hull area; 1.0 for a convex shape."""
    from skimage.measure import label, regionprops

    mask = np.asarray(mask, bool)
    if not mask.any():
        return float("nan")
    props = regionprops(label(mask))
    area = sum(p.area for p in props)
    hull = sum(p.convex_area for p in props)
    return float(area / hull) if hull else float("nan")


def is_regular(mask: np.ndarray, min_solidity: float = 0.95) -> bool:
    """Prespecified morphology rule: one component and solidity >= 0.95."""
    from skimage.measure import label

    mask = np.asarray(mask, bool)
    n_components = label(mask).max()
    return bool(n_components == 1 and solidity(mask) >= min_solidity)


def all_metrics(pred: np.ndarray, ref: np.ndarray, nsd_tolerance: float = 3.0):
    return {
        "dice": dice(pred, ref),
        "iou": iou(pred, ref),
        "nsd": nsd(pred, ref, nsd_tolerance),
        "hd95": hd95(pred, ref),
        "abs_rel_area_error": abs_relative_area_error(pred, ref),
        "equiv_diameter_error": equivalent_diameter_error(pred, ref),
    }
