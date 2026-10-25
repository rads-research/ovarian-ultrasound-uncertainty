#!/usr/bin/env python3
"""Generate SYNTHETIC data for smoke-testing the plotting code.

================= THIS IS NOT RESEARCH DATA =================
The numbers produced here are fabricated. They reproduce only the
published marginal counts and means so that the figure code can be
exercised end to end before the real per-image results are available.
Output goes to results/_smoke/ and must never be used for analysis,
reported, or committed as results.
=============================================================

    python scripts/make_smoke_data.py
    python scripts/make_figures.py --per-image results/_smoke/per_image.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import (  # noqa: E402
    CALIBRATION_BINS, MAJOR_FAILURE_DICE, REFERRAL_THRESHOLD, RESULTS_DIR,
)

OUT = RESULTS_DIR / "_smoke"

# cohort -> (n, flagged failures, silent failures, false-positive referrals,
#            target mean Dice, target mean disagreement)
SPEC = {
    "2d_validation": (204, 6, 1, 9, 0.875, 0.112),
    "2d_test":       (469, 9, 13, 24, 0.864, 0.130),
    "ceus":          (170, 18, 9, 29, 0.722, 0.256),
}


def _fit_mean(values: np.ndarray, target: float, lo: float, hi: float):
    """Shift values to hit `target` mean while staying inside (lo, hi)."""
    for _ in range(200):
        err = target - values.mean()
        if abs(err) < 1e-6:
            break
        values = np.clip(values + err, lo, hi)
    return values


def build(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for cohort, (n, flagged, silent, fp, mdice, mdis) in SPEC.items():
        correct = n - flagged - silent - fp
        assert correct > 0, cohort

        dice, dis, kind = [], [], []

        # failures referred for review: low Dice, high disagreement
        dice += list(rng.uniform(0.00, 0.49, flagged))
        dis += list(rng.uniform(REFERRAL_THRESHOLD + 0.01, 0.95, flagged))
        kind += ["flagged"] * flagged
        # silent failures: low Dice, low disagreement
        dice += list(rng.uniform(0.00, 0.49, silent))
        dis += list(rng.uniform(0.02, REFERRAL_THRESHOLD - 0.01, silent))
        kind += ["silent"] * silent
        # false-positive referrals: acceptable Dice, high disagreement
        dice += list(rng.uniform(0.51, 0.88, fp))
        dis += list(rng.uniform(REFERRAL_THRESHOLD + 0.01, 0.80, fp))
        kind += ["false_positive"] * fp
        # correctly retained: high Dice, low disagreement
        d = rng.beta(7.5, 1.6, correct) * 0.49 + 0.51
        dice += list(d)
        dis += list(rng.beta(1.8, 9.0, correct) * (REFERRAL_THRESHOLD - 0.02))
        kind += ["correct"] * correct

        dice = np.asarray(dice)
        dis = np.asarray(dis)
        kind = np.asarray(kind)

        # Nudge the two marginal means onto target without crossing any
        # threshold, so the counts above are preserved exactly.
        keep = kind == "correct"
        dice[keep] = _fit_mean(
            dice[keep], (mdice * n - dice[~keep].sum()) / keep.sum(), 0.51, 1.0)
        dis[keep] = _fit_mean(
            dis[keep], max((mdis * n - dis[~keep].sum()) / keep.sum(), 0.01),
            0.001, REFERRAL_THRESHOLD - 0.001)

        ref_area = rng.uniform(0.01, 0.45, n)
        rows.append(pd.DataFrame({
            "image_id": [f"{cohort}_{i:04d}" for i in range(n)],
            "cohort": cohort,
            "dice": dice,
            "disagreement": dis,
            "mean_entropy": dis * rng.uniform(0.55, 0.95, n),
            "ref_area_frac": ref_area,
            "pred_area_frac": ref_area * rng.uniform(0.6, 1.5, n),
            "group_id": [f"{cohort}_g{i:04d}" for i in range(n)],
            "morphology": rng.choice(["regular", "irregular"], n),
            "size_tertile": rng.choice(["small", "medium", "large"], n),
        }))
    return pd.concat(rows, ignore_index=True)


def build_reliability(rng: np.random.Generator) -> pd.DataFrame:
    edges = np.linspace(0, 1, CALIBRATION_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    rows = []
    for cohort, bias in (("2d_test", 0.03), ("ceus", 0.11)):
        for source in ["ensemble"] + [f"member_{i:02d}" for i in range(15)]:
            b = bias if source == "ensemble" else bias * rng.uniform(1.4, 2.4)
            obs = np.clip(centres - b * np.sin(np.pi * centres), 0, 1)
            rows.append(pd.DataFrame({
                "cohort": cohort, "source": source, "region": "lesion",
                "bin_index": np.arange(CALIBRATION_BINS),
                "mean_pred": centres, "observed_frequency": obs,
                "n_pixels": rng.integers(5_000, 200_000, CALIBRATION_BINS),
            }))
    return pd.concat(rows, ignore_index=True)


def build_example_images(df: pd.DataFrame, rng: np.random.Generator) -> int:
    """Fake ultrasound frames and masks so Supplementary Fig. 2 can render."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = OUT / "silent_failures"
    out.mkdir(parents=True, exist_ok=True)
    silent = df[(df["cohort"] == "ceus")
                & (df["dice"] < MAJOR_FAILURE_DICE)
                & (df["disagreement"] < REFERRAL_THRESHOLD)].sort_values("dice")

    size = 256
    yy, xx = np.mgrid[0:size, 0:size]
    for _, row in silent.iterrows():
        frame = rng.normal(0.35, 0.11, (size, size))
        frame = np.clip(frame + 0.25 * np.exp(-((yy - 128) ** 2) / 2e4), 0, 1)

        r_ref = rng.integers(34, 56)
        cy, cx = rng.integers(90, 165, 2)
        ref = ((yy - cy) ** 2 + (xx - cx) ** 2) < r_ref ** 2

        # Displace and resize the prediction to approximate the stated Dice.
        shift = int((1 - row["dice"]) * 70)
        r_pred = int(r_ref * rng.uniform(0.55, 1.5))
        pred = ((yy - cy - shift) ** 2 + (xx - cx + shift // 2) ** 2) < r_pred ** 2

        stem = out / str(row["image_id"])
        plt.imsave(f"{stem}_image.png", frame, cmap="gray", vmin=0, vmax=1)
        plt.imsave(f"{stem}_ref.png", ref.astype(float), cmap="gray")
        plt.imsave(f"{stem}_pred.png", pred.astype(float), cmap="gray")
    return len(silent)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260101)

    df = build(rng)
    df.to_csv(OUT / "per_image.csv", index=False)
    build_reliability(rng).to_csv(OUT / "reliability.csv", index=False)
    n_img = build_example_images(df, rng)
    print(f"  wrote {n_img} synthetic silent-failure example image sets")

    print(f"SYNTHETIC smoke data written to {OUT}/")
    print(f"{'cohort':15s} {'n':>4s} {'fail':>5s} {'ref':>5s} {'silent':>7s} "
          f"{'meanDice':>9s} {'meanDis':>8s}")
    for cohort, sub in df.groupby("cohort", sort=False):
        fail = sub["dice"] < MAJOR_FAILURE_DICE
        ref = sub["disagreement"] >= REFERRAL_THRESHOLD
        print(f"{cohort:15s} {len(sub):4d} {int(fail.sum()):5d} "
              f"{int(ref.sum()):5d} {int((fail & ~ref).sum()):7d} "
              f"{sub['dice'].mean():9.4f} {sub['disagreement'].mean():8.4f}")
    print("\nReminder: these values are fabricated. Replace with real results.")


if __name__ == "__main__":
    main()
