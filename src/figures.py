"""Figure generation for the manuscript.

All six figures are produced from two inputs:

  results/per_image.csv       one row per image (see results/SCHEMA.md)
  results/reliability.csv     pixel-level calibration bins

Risk-coverage curves, ROC and precision-recall curves are derived from
per_image.csv rather than stored separately, so they cannot drift out of
step with the scatter plots.

Supplementary Figure 2 additionally reads example images and masks from
results/silent_failures/.

Styling rules, applied throughout:
  * Okabe-Ito colour-vision-safe palette.
  * Colour never carries information alone; every series also differs by
    marker and line style, and panels are labelled in text.
  * Serif typeface matching the manuscript (Times New Roman where present).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    COHORT_LABELS, COHORT_ORDER, COHORT_STYLE, MAJOR_FAILURE_DICE,
    OKABE_ITO, REFERRAL_THRESHOLD, RESULTS_DIR, WIDTH_DOUBLE, WIDTH_ONEHALF,
    WIDTH_SINGLE, apply_style, save_figure,
)

TEST_COHORTS = ("2d_test", "ceus")


# ==========================================================================
# Data loading
# ==========================================================================

def load_per_image(path: Path | None = None) -> pd.DataFrame:
    path = path or RESULTS_DIR / "per_image.csv"
    df = pd.read_csv(path)
    required = {"image_id", "cohort", "dice", "disagreement"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    unknown = set(df["cohort"]) - set(COHORT_ORDER)
    if unknown:
        raise ValueError(
            f"Unrecognised cohort label(s) {sorted(unknown)}. "
            f"Expected one of {COHORT_ORDER}."
        )
    df["failure"] = df["dice"] < MAJOR_FAILURE_DICE
    df["referred"] = df["disagreement"] >= REFERRAL_THRESHOLD
    df["silent"] = df["failure"] & ~df["referred"]
    return df


def risk_coverage(uncertainty: np.ndarray, risk: np.ndarray):
    """Risk among retained images as coverage falls from 1 to 1/n.

    Images are excluded in order of decreasing uncertainty, so at coverage c
    the retained set is the ceil(c*n) images with the lowest uncertainty.
    Returns (coverage, mean risk among retained, AURC, AURC under random
    referral). Under random referral the expected risk is the base rate at
    every coverage, so its AURC is the base rate itself.
    """
    order = np.argsort(uncertainty, kind="mergesort")
    risk_sorted = np.asarray(risk, dtype=float)[order]
    n = len(risk_sorted)
    k = np.arange(1, n + 1)
    cumulative = np.cumsum(risk_sorted) / k
    coverage = k / n
    return coverage, cumulative, float(cumulative.mean()), float(risk_sorted.mean())


def roc_curve(score: np.ndarray, label: np.ndarray):
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    y = np.asarray(label, dtype=bool)[order]
    tp = np.concatenate([[0], np.cumsum(y)])
    fp = np.concatenate([[0], np.cumsum(~y)])
    if tp[-1] == 0 or fp[-1] == 0:
        raise ValueError("ROC needs both positive and negative cases.")
    tpr, fpr = tp / tp[-1], fp / fp[-1]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def pr_curve(score: np.ndarray, label: np.ndarray):
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    y = np.asarray(label, dtype=bool)[order]
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    precision = tp / k
    recall = tp / tp[-1]
    ap = float(np.sum(np.diff(np.concatenate([[0], recall])) * precision))
    return recall, precision, ap


def _panel_label(ax, text, dx=-0.17, dy=1.04):
    ax.text(dx, dy, text, transform=ax.transAxes, fontweight="bold",
            fontsize=plt.rcParams["font.size"] + 1, va="bottom", ha="left")


# ==========================================================================
# Figure 1 - Dice by cohort
# ==========================================================================

def figure1(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 2.5))
    rng = np.random.default_rng(0)

    for i, cohort in enumerate(COHORT_ORDER):
        sub = df[df["cohort"] == cohort]
        if sub.empty:
            continue
        style = COHORT_STYLE[cohort]
        jitter = rng.uniform(-0.17, 0.17, len(sub))
        ax.scatter(i + jitter, sub["dice"], s=4, marker=style["marker"],
                   facecolors="none", edgecolors=style["color"],
                   linewidths=0.4, alpha=0.65, zorder=2,
                   rasterized=len(sub) > 400)

        q1, med, q3 = np.percentile(sub["dice"], [25, 50, 75])
        ax.plot([i - 0.33, i + 0.33], [med, med], color="black",
                lw=1.3, zorder=4, solid_capstyle="butt")
        ax.plot([i, i], [q1, q3], color="black", lw=0.7, zorder=3)
        ax.text(i, 1.055, f"n={len(sub)}", ha="center", va="bottom",
                fontsize=plt.rcParams["font.size"] - 1)

    # The dashed line is identified in the caption. An inline label here sits
    # on top of the failure points it is describing, so it is omitted.
    ax.axhline(MAJOR_FAILURE_DICE, color=OKABE_ITO["black"], ls="--",
               lw=0.7, zorder=1)

    ax.set_xticks(range(len(COHORT_ORDER)))
    ax.set_xticklabels([COHORT_LABELS[c] for c in COHORT_ORDER])
    ax.set_ylabel("Dice similarity coefficient")
    ax.set_ylim(-0.03, 1.09)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(-0.6, len(COHORT_ORDER) - 0.4)
    fig.tight_layout()
    return fig


# ==========================================================================
# Figure 2 - uncertainty against Dice
# ==========================================================================

def figure2(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_ONEHALF, 2.6),
                             sharex=True, sharey=True)

    for ax, cohort, tag in zip(axes, TEST_COHORTS, ("A", "B")):
        sub = df[df["cohort"] == cohort]
        style = COHORT_STYLE[cohort]

        # Silent-failure quadrant, drawn in data coordinates so it ends
        # exactly at the two thresholds regardless of axis limits.
        ax.add_patch(plt.Rectangle(
            (0.0, 0.0), REFERRAL_THRESHOLD, MAJOR_FAILURE_DICE,
            facecolor=OKABE_ITO["yellow"], alpha=0.30, edgecolor="none",
            zorder=0))

        ok = sub[~sub["failure"]]
        flagged = sub[sub["failure"] & sub["referred"]]
        silent = sub[sub["silent"]]

        ax.scatter(ok["disagreement"], ok["dice"], s=4,
                   marker=style["marker"], facecolors="none",
                   edgecolors=style["color"], linewidths=0.4, alpha=0.55,
                   zorder=2, label="No major failure",
                   rasterized=len(ok) > 400)
        ax.scatter(flagged["disagreement"], flagged["dice"], s=13,
                   marker="D", facecolors="none",
                   edgecolors=OKABE_ITO["bluish_green"], linewidths=0.8,
                   zorder=3, label="Major failure, referred")
        ax.scatter(silent["disagreement"], silent["dice"], s=20, marker="x",
                   color=OKABE_ITO["black"], linewidths=0.9, zorder=4,
                   label="Silent failure")

        ax.axhline(MAJOR_FAILURE_DICE, color="black", ls="--", lw=0.7, zorder=1)
        ax.axvline(REFERRAL_THRESHOLD, color="black", ls=":", lw=0.7, zorder=1)
        ax.set_title(COHORT_LABELS[cohort], pad=3)
        ax.set_xlabel("Ensemble disagreement")
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(np.arange(0, 0.81, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        _panel_label(ax, tag, dx=-0.13)

    axes[0].set_ylabel("Dice similarity coefficient")
    # The shaded quadrant is identified in the caption. An in-plot label sits
    # on the very points it describes, so it is omitted.
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    return fig


# ==========================================================================
# Figure 3 - risk-coverage
# ==========================================================================

def figure3(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_ONEHALF, 2.6))
    summary = {}

    for cohort in TEST_COHORTS:
        sub = df[df["cohort"] == cohort]
        style = COHORT_STYLE[cohort]
        u = sub["disagreement"].to_numpy()

        cov, risk, aurc, rand = risk_coverage(u, sub["failure"].to_numpy())
        axes[0].plot(cov, risk, color=style["color"], ls=style["ls"],
                     label=COHORT_LABELS[cohort])
        # The random-referral reference is a horizontal line at the base rate,
        # which runs along the right-hand end of the curve itself. Drawing it
        # obscures the curve, so the comparison is given in the caption and
        # the text instead.

        cov_d, mean_dice, _, _ = risk_coverage(u, sub["dice"].to_numpy())
        axes[1].plot(cov_d, mean_dice, color=style["color"], ls=style["ls"],
                     label=COHORT_LABELS[cohort])

        summary[cohort] = {"aurc_failure": aurc, "aurc_random": rand}

    axes[0].set_ylabel("Major failure rate among retained images")
    axes[1].set_ylabel("Mean Dice among retained images")
    for ax, tag in zip(axes, ("A", "B")):
        ax.set_xlabel("Coverage (proportion retained)")
        ax.set_xlim(0.25, 1.01)
        ax.invert_xaxis()
        _panel_label(ax, tag, dx=-0.20)
    axes[0].set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    return fig, summary


# ==========================================================================
# Supplementary Figure 1 - ROC and precision-recall
# ==========================================================================

def supp_figure1(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_ONEHALF, 2.6))
    stats = {}

    for cohort in TEST_COHORTS:
        sub = df[df["cohort"] == cohort]
        style = COHORT_STYLE[cohort]
        score = sub["disagreement"].to_numpy()
        label = sub["failure"].to_numpy()

        fpr, tpr, auroc = roc_curve(score, label)
        axes[0].plot(fpr, tpr, color=style["color"], ls=style["ls"],
                     label=COHORT_LABELS[cohort])

        rec, prec, ap = pr_curve(score, label)
        prevalence = label.mean()
        axes[1].plot(rec, prec, color=style["color"], ls=style["ls"],
                     label=COHORT_LABELS[cohort])
        # A horizontal chance line at the prevalence runs along the
        # precision-recall curve itself at low recall and obscures it. The
        # prevalence of each cohort is given in the caption instead.
        stats[cohort] = {"auroc": auroc, "auprc": ap, "prevalence": prevalence}

    axes[0].plot([0, 1], [0, 1], color=OKABE_ITO["grey"], ls=(0, (1, 2)),
                 lw=0.7)
    axes[0].set_xlabel("1 - specificity")
    axes[0].set_ylabel("Sensitivity")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    for ax, tag in zip(axes, ("A", "B")):
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right" if tag == "A" else "upper right")
        _panel_label(ax, tag, dx=-0.20)
    fig.tight_layout()
    return fig, stats


# ==========================================================================
# Supplementary Figure 2 - example silent failures
# ==========================================================================

def supp_figure2(df: pd.DataFrame, image_dir: Path | None = None,
                 n_examples: int = 3, image_ids: list[str] | None = None):
    """Example silent failures with reference and predicted contours.

    Expects, for each chosen image_id, three files in image_dir:
        <image_id>_image.png   grayscale frame
        <image_id>_ref.png     binary reference mask
        <image_id>_pred.png    binary ensemble mask

    By default the examples span the observed range rather than the worst
    cases, so that both failure patterns are visible: picking only the
    lowest-Dice cases yields panels that are all near zero and look
    identical. Pass image_ids to choose explicitly.
    """
    from matplotlib import colors as mcolors

    image_dir = image_dir or RESULTS_DIR / "silent_failures"
    silent = df[(df["cohort"] == "ceus") & df["silent"]].sort_values("dice")
    if silent.empty:
        raise ValueError("No silent failures found in the CEUS cohort.")

    if image_ids is None:
        idx = np.linspace(0, len(silent) - 1, n_examples).round().astype(int)
        chosen = silent.iloc[idx]
    else:
        chosen = silent.set_index("image_id").loc[image_ids].reset_index()

    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE / n + 0.3))
    axes = np.atleast_1d(axes)

    for ax, (_, row) in zip(axes, chosen.iterrows()):
        # image_id may be qualified by cohort ("ceus/23") because stems repeat
        # across the two MMOTU subsets; the files are named by stem alone.
        file_stem = str(row["image_id"]).rsplit("/", 1)[-1]
        stem = image_dir / file_stem
        img = plt.imread(f"{stem}_image.png")
        ref = plt.imread(f"{stem}_ref.png")
        pred = plt.imread(f"{stem}_pred.png")
        def _to_2d(a):
            """Accept grayscale, RGB or RGBA; return a 2D float array."""
            a = np.asarray(a)
            if a.ndim == 3:
                a = a[..., :3].mean(axis=2)
            return np.squeeze(a).astype(float)

        img = _to_2d(img)
        ref = (_to_2d(ref) > 0.5).astype(float)
        pred = (_to_2d(pred) > 0.5).astype(float)
        if ref.shape != img.shape or pred.shape != img.shape:
            raise ValueError(
                f"{row['image_id']}: mask shape {ref.shape}/{pred.shape} "
                f"does not match image shape {img.shape}.")

        ax.imshow(img, cmap="gray", interpolation="nearest")
        # Solid reference, dashed prediction: distinguishable in greyscale.
        ax.contour(ref, levels=[0.5], colors=[OKABE_ITO["sky_blue"]],
                   linewidths=1.1, linestyles="solid")
        ax.contour(pred, levels=[0.5], colors=[OKABE_ITO["vermillion"]],
                   linewidths=1.1, linestyles="dashed")
        ax.set_title(
            f"Dice {row['dice']:.2f}\ndisagreement {row['disagreement']:.2f}",
            pad=3, fontsize=plt.rcParams["font.size"] - 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

    handles = [
        plt.Line2D([], [], color=OKABE_ITO["sky_blue"], ls="-", lw=1.1,
                   label="Expert reference"),
        plt.Line2D([], [], color=OKABE_ITO["vermillion"], ls="--", lw=1.1,
                   label="Ensemble prediction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    return fig, chosen


# ==========================================================================
# Supplementary Figure 3 - reliability diagrams
# ==========================================================================

def supp_figure3(path: Path | None = None):
    """Reliability diagrams, from the run's calibration table.

    Expected columns (as written by the calibration stage):
        cohort              '2D internal test' | 'CEUS stress test'
        scope               'ensemble, all pixels' | 'ensemble, lesion ROI'
                            | 'single model, mean of 15'
        bin_low, bin_high   bin edges
        mean_confidence     mean predicted probability in the bin
        observed_frequency  observed foreground frequency in the bin

    One panel per cohort, one curve per scope. Single-model values exist
    only over all pixels, so three curves are drawn rather than the 15
    individual member curves.
    """
    path = path or RESULTS_DIR / "reliability.csv"
    rel = pd.read_csv(path)

    need = {"cohort", "scope", "mean_confidence", "observed_frequency"}
    missing = need - set(rel.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")

    def norm(v):
        v = str(v).lower()
        if "ceus" in v or "3d" in v:
            return "ceus"
        if "test" in v or "internal" in v or "2d" in v:
            return "2d_test"
        return v

    rel = rel.copy()
    rel["coh"] = rel["cohort"].map(norm)
    if "bin_low" in rel.columns:
        rel = rel.sort_values("bin_low")

    SCOPE_STYLE = {
        "ensemble, lesion roi":       dict(color=OKABE_ITO["blue"], ls="-",
                                           marker="o", label="Ensemble, lesion region"),
        "ensemble, all pixels":       dict(color=OKABE_ITO["bluish_green"], ls="--",
                                           marker="s", label="Ensemble, all pixels"),
        "single model, mean of 15":   dict(color=OKABE_ITO["grey"], ls=":",
                                           marker="^", label="Single model, mean of 15"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_ONEHALF, 2.7),
                             sharex=True, sharey=True)
    for ax, cohort, tag in zip(axes, TEST_COHORTS, ("A", "B")):
        sub = rel[rel["coh"] == cohort]
        if sub.empty:
            raise ValueError(f"No calibration rows for cohort '{cohort}'.")
        ax.plot([0, 1], [0, 1], color="black", ls="--", lw=0.7, zorder=1)
        for scope, grp in sub.groupby("scope"):
            st = SCOPE_STYLE.get(str(scope).strip().lower())
            if st is None:
                continue
            ax.plot(grp["mean_confidence"], grp["observed_frequency"],
                    color=st["color"], ls=st["ls"], marker=st["marker"],
                    markersize=2.6, lw=1.1, zorder=3, label=st["label"])
        ax.set_title(COHORT_LABELS[cohort], pad=3)
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(np.arange(0, 0.81, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        _panel_label(ax, tag, dx=-0.13)

    axes[0].set_ylabel("Observed foreground frequency")
    handles, labels = axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    h2.append(plt.Line2D([], [], color="black", ls="--", lw=0.7))
    l2.append("Perfect calibration")
    fig.legend(h2, l2, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.19))
    fig.tight_layout()
    return fig


# ==========================================================================
# Layout check
# ==========================================================================

def check_overlaps(fig, tol: float = 0.0) -> list[str]:
    """Report text elements whose bounding boxes intersect.

    The eye misses small collisions at print size; this catches them before
    submission. Returns a list of human-readable collision descriptions.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = []
    for ax in fig.axes:
        # Matplotlib keeps tick-label objects for ticks outside the view;
        # they never render, so comparing them yields false positives.
        # Tick locations and labels are parallel, so filter by location.
        candidates = list(ax.texts)
        for locs, labels, lim in (
            (ax.get_xticks(), ax.get_xticklabels(), ax.get_xlim()),
            (ax.get_yticks(), ax.get_yticklabels(), ax.get_ylim()),
        ):
            lo, hi = sorted(lim)
            for loc, lab in zip(locs, labels):
                if lo <= loc <= hi:
                    candidates.append(lab)

        for t in candidates:
            if not t.get_text().strip() or not t.get_visible():
                continue
            items.append((t.get_text(), t.get_window_extent(renderer)))
        if ax.get_legend() is not None:
            items.append(("<legend>", ax.get_legend().get_window_extent(renderer)))
    collisions = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (ta, ba), (tb, bb) = items[i], items[j]
            if ba.overlaps(bb):
                ox = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
                oy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
                if ox > tol and oy > tol:
                    collisions.append(
                        f"{ta[:28]!r} overlaps {tb[:28]!r} "
                        f"({ox:.1f} x {oy:.1f} px)")

    # Text sitting on top of plotted data is just as unreadable as text on
    # text, and the eye misses it at print size. An annotation with an opaque
    # background patch is exempt: it is legible over whatever lies beneath.
    for ax in fig.axes:
        for t in ax.texts:
            if not t.get_text().strip() or not t.get_visible():
                continue
            patch = t.get_bbox_patch()
            if patch is not None and patch.get_alpha() not in (None, 0):
                continue
            bb = t.get_window_extent(renderer)
            for coll in ax.collections:
                try:
                    pts = ax.transData.transform(coll.get_offsets())
                except Exception:
                    continue
                hits = sum(1 for x, y in pts
                           if bb.x0 <= x <= bb.x1 and bb.y0 <= y <= bb.y1)
                if hits:
                    collisions.append(
                        f"{t.get_text()[:28]!r} sits on {hits} plotted point(s)")
    return collisions
