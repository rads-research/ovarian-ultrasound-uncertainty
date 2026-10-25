#!/usr/bin/env python3
"""Generate every manuscript figure.

    python scripts/make_figures.py                 # all figures
    python scripts/make_figures.py --only 1 2      # main figures 1 and 2
    python scripts/make_figures.py --formats pdf   # draft PDFs only

Outputs land in figures/ as 600 dpi LZW-compressed TIFF plus vector PDF.
Each figure is checked for overlapping text before it is written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import figures as F  # noqa: E402
from config import (  # noqa: E402
    MANUSCRIPT_COUNTS, MANUSCRIPT_VALUES, OUTPUT_FORMATS, apply_style,
    save_figure,
)


def verify_against_manuscript(df) -> list[str]:
    """Recompute every value the manuscript prints. Empty list == agreement.

    A figure drawn from data that contradicts the text is worse than no
    figure, so this runs before anything is written and aborts the build.
    """
    import numpy as np
    problems = []

    for cohort, (n, fail, ref, silent) in MANUSCRIPT_COUNTS.items():
        sub = df[df["cohort"] == cohort]
        got = (len(sub), int(sub["failure"].sum()), int(sub["referred"].sum()),
               int(sub["silent"].sum()))
        if got != (n, fail, ref, silent):
            problems.append(
                f"{cohort}: n/failures/referred/silent = {got}, "
                f"manuscript says {(n, fail, ref, silent)}")

    computed = {}
    for cohort in df["cohort"].unique():
        sub = df[df["cohort"] == cohort]
        u = sub["disagreement"].to_numpy()
        f = sub["failure"].to_numpy()
        computed[(cohort, "mean_dice")] = float(sub["dice"].mean())
        if f.any() and not f.all():
            computed[(cohort, "auroc")] = F.roc_curve(u, f)[2]
            computed[(cohort, "auprc")] = F.pr_curve(u, f)[2]
            _, _, aurc, rand = F.risk_coverage(u, f)
            computed[(cohort, "aurc")] = aurc
            computed[(cohort, "aurc_random")] = rand

    for key, (printed, dp) in MANUSCRIPT_VALUES.items():
        if key not in computed:
            continue
        got = round(computed[key], dp)
        if abs(got - printed) > 10 ** (-dp) / 2:
            problems.append(
                f"{key[0]} {key[1]}: computes to {computed[key]:.5f} "
                f"(= {got:.{dp}f}), manuscript prints {printed:.{dp}f}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of 1 2 3 s1 s2 s3")
    ap.add_argument("--formats", nargs="*", default=list(OUTPUT_FORMATS))
    ap.add_argument("--per-image", type=Path, default=None)
    ap.add_argument("--reliability", type=Path, default=None)
    ap.add_argument("--images", type=Path, default=None,
                    help="directory of silent-failure example images")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="build even if the data disagrees with the "
                         "manuscript. For smoke-testing the plotting code "
                         "with synthetic data ONLY -- never for submission.")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any text overlap is detected")
    args = ap.parse_args()

    font = apply_style()
    print(f"Serif font in use: {font}")

    df = F.load_per_image(args.per_image)
    print(f"Loaded {len(df)} images across {df['cohort'].nunique()} cohorts")
    for cohort, sub in df.groupby("cohort"):
        print(f"  {cohort:14s} n={len(sub):4d}  failures={int(sub['failure'].sum()):3d}"
              f"  referred={int(sub['referred'].sum()):3d}"
              f"  silent={int(sub['silent'].sum()):3d}")

    problems = verify_against_manuscript(df)
    if problems and args.allow_unverified:
        print("\n*** WARNING: --allow-unverified is set. The data below does "
              "NOT match the manuscript.")
        print("*** The figures produced are NOT suitable for submission.\n")
        for p in problems:
            print(f"  - {p}")
        print()
        problems = []
    if problems:
        print("\nABORTED: the data disagrees with the manuscript.\n")
        for p in problems:
            print(f"  - {p}")
        print("\nNo figure was written. Either the wrong per-image file was "
              "supplied, or the manuscript needs correcting.")
        return 2
    print(f"Verified {len(MANUSCRIPT_VALUES)} manuscript values and "
          f"{len(MANUSCRIPT_COUNTS)} cohort count sets: all agree.\n")

    want = set(args.only) if args.only else {"1", "2", "3", "s1", "s2", "s3"}
    overlaps_found = False
    built: list[str] = []

    def emit(fig, stem: str) -> None:
        nonlocal overlaps_found
        problems = F.check_overlaps(fig)
        if problems:
            overlaps_found = True
            print(f"  ! {stem}: {len(problems)} text overlap(s)")
            for p in problems:
                print(f"      {p}")
        for path in save_figure(fig, stem, formats=args.formats):
            print(f"  wrote {path.relative_to(path.parent.parent)}")
        plt.close(fig)
        built.append(stem)

    if "1" in want:
        emit(F.figure1(df), "Fig1_dice_by_cohort")
    if "2" in want:
        emit(F.figure2(df), "Fig2_uncertainty_vs_dice")
    if "3" in want:
        fig, summary = F.figure3(df)
        emit(fig, "Fig3_risk_coverage")
        for cohort, s in summary.items():
            print(f"      {cohort}: AURC(failure)={s['aurc_failure']:.4f} "
                  f"vs random {s['aurc_random']:.4f}")
    if "s1" in want:
        fig, stats = F.supp_figure1(df)
        emit(fig, "SuppFig1_roc_pr")
        for cohort, s in stats.items():
            print(f"      {cohort}: AUROC={s['auroc']:.4f} "
                  f"AUPRC={s['auprc']:.4f} prevalence={s['prevalence']:.4f}")
    if "s2" in want:
        try:
            fig, chosen = F.supp_figure2(df, image_dir=args.images)
            emit(fig, "SuppFig2_silent_failures")
            print("      examples:", ", ".join(map(str, chosen["image_id"])))
        except FileNotFoundError as exc:
            print(f"  - SuppFig2 skipped: {exc}")
    if "s3" in want:
        try:
            emit(F.supp_figure3(args.reliability), "SuppFig3_reliability")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  - SuppFig3 skipped: {exc}")

    print(f"\nBuilt {len(built)} figure(s).")
    if overlaps_found:
        print("Text overlaps were detected; inspect the figures above.")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
