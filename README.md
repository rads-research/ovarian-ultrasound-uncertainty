# Uncertainty-based referral for ovarian lesion segmentation under cross-modality shift

Analysis code for a study of whether ensemble disagreement identifies
segmentation failures when a model developed on conventional
two-dimensional (2D) ovarian ultrasound is applied unchanged to
contrast-enhanced ultrasound (CEUS).

> **Anonymised for peer review.** Author, institution and funding details
> are withheld and will be added after the review process concludes.

## What this repository contains

| Path | Purpose |
|---|---|
| `src/metrics.py` | Dice, IoU, NSD, HD95, area and equivalent-diameter error, morphology rule |
| `src/uncertainty.py` | Ensemble disagreement (primary) and the comparator measures; calibration |
| `src/stats.py` | Clustered bootstrap, Holm–Bonferroni, Benjamini–Hochberg, design effects, risk-coverage |
| `src/figures.py` | All six manuscript figures |
| `src/config.py` | Prespecified constants, training hyperparameters, augmentation, cohort definitions, figure styling |
| `scripts/make_figures.py` | Figure driver |
| `scripts/make_smoke_data.py` | Synthetic data for testing the plotting code only |
| `prespecification/` | The locked analysis plan and its SHA-256 hash |

## The prespecification lock

Every analytical decision — ensemble membership and combination rule,
segmentation threshold, failure and silent-failure definitions,
uncertainty measures, referral threshold, subgroup definitions and the
statistical analysis plan — was recorded before any CEUS image was read.
The analysis verifies the hash of that file before loading CEUS data and
refuses to continue if it does not match:

```bash
python prespecification/verify_hash.py
```

Four analyses (equivalent-diameter error, the comparison of uncertainty
measures, the ensemble-size and composition ablations, and calibration)
were added after the locked results were examined. They are labelled
exploratory in the code and in the manuscript, and are not corrected for
multiplicity.

## Requirements

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.13.16 with PyTorch 2.11.0 (CUDA 13.0) on a single
NVIDIA A100.
Figure generation needs no GPU.

## Data

The MMOTU ovarian ultrasound dataset is redistributed by its original
authors, not here. See `data/README.md` for how to obtain and arrange it.
In brief: 1,469 conventional 2D images and 170 CEUS images, with
expert pixel-level annotations, from a single centre.

We use the dataset authors' own 2D partition (1,000 development / 469
held-out test), which they report as disjoint by patient. All 170 CEUS
images form the cross-modality cohort. No CEUS image contributes to
training, model selection, calibration or threshold selection.

## Reproducing the analysis

```bash
# Statistics: six prespecified hypotheses, sensitivity and exploratory
python scripts/run_analysis.py          # not included; see note below

# Figures
python scripts/make_figures.py
```

**What is and is not included.** This repository contains the analysis and
figure code, the prespecification, and the full configuration. The training
and inference drivers (`train.py`, `predict.py`, `run_analysis.py`) are not
included, because they depend on infrastructure specific to the compute
environment used. Every setting they consumed is recorded in
`src/config.py`, and `src/metrics.py`, `src/uncertainty.py` and
`src/stats.py` contain the exact definitions of every quantity reported.
Given `results/per_image.csv`, `scripts/make_figures.py` reproduces every
figure on CPU in under a minute.


### Checking the plotting code without the real data

```bash
python scripts/make_smoke_data.py
python scripts/make_figures.py --allow-unverified \
    --per-image   results/_smoke/per_image.csv \
    --reliability results/_smoke/reliability.csv \
    --images      results/_smoke/silent_failures
```

`--allow-unverified` is required here and only here. By default
`make_figures.py` recomputes every value the manuscript reports and
refuses to draw anything that disagrees with it; synthetic data does not
agree, by construction.

The values this generates are **synthetic**. They reproduce the published
marginal counts and means only so that the figure code can be exercised
end to end. They are not research data and must never be reported.

## Training configuration

Optimiser: AdamW throughout, with a 3-epoch linear warm-up followed by
cosine annealing, up to 80 epochs, batch size 8 (reduced automatically to a
minimum of 2 where GPU memory required), combined Dice and cross-entropy
loss, retaining for each model the epoch of highest validation Dice.

| Architecture | Learning rate | Weight decay |
|---|---|---|
| U-Net, ImageNet-pretrained ResNet-34 encoder | 3 × 10⁻⁴ | 1 × 10⁻⁴ |
| nnU-Net default 2D architecture | 1 × 10⁻³ | 1 × 10⁻⁵ |
| SegFormer, MiT-B2 encoder | 6 × 10⁻⁵ | 1 × 10⁻² |

| Augmentation | Parameters | Probability |
|---|---|---|
| Horizontal flip | — | 0.50 |
| Translation | ±6.25% of image size | 0.50 |
| Scaling | 0.85–1.15× | 0.50 |
| Rotation | ±15° | 0.50 |
| Brightness / contrast | ±20% each | 0.50 |
| Gaussian noise | standard deviation 0.02–0.12 | 0.20 |
| Gaussian blur | kernel size 3–5 | 0.15 |
| Elastic deformation | α = 30, σ = 6 | 0.15 |

No vertical flip was applied. Images were resized to 384 × 384 with
aspect-ratio-preserving padding, scaled by 1/255 with no per-image or
per-cohort z-scoring, normalised with fixed ImageNet statistics, and
replicated to three channels.

## Figure conventions

- Okabe–Ito colour-vision-safe palette throughout.
- Colour never carries information on its own: every series also differs
  by marker and line style, and panels are labelled in text.
- Serif typeface matching the manuscript, falling back to a
  metric-compatible substitute where Times New Roman is unavailable.
- Output at 600 dpi as LZW-compressed TIFF plus vector PDF, sized to
  journal column widths (90 / 140 / 190 mm).
- `make_figures.py` checks every figure for overlapping text before
  writing it; `--strict` makes a collision a non-zero exit.

## Expected inputs

`results/per_image.csv` and `results/reliability.csv` are the only inputs
the analysis and figure code need. Their schemas are in
`results/SCHEMA.md`. Risk-coverage, ROC and precision-recall curves are
derived from the per-image table rather than stored separately, so they
cannot drift out of step with the scatter plots.

## Notes on the statistics

- Confidence intervals resample **image groups**, not images, so
  near-duplicate frames cannot be split across a replicate. With 2,000
  replicates the minimum attainable P value is 0.001; a P at that floor
  means no replicate crossed the null, not a precisely estimated small
  value.
- The design-effect analysis uses the **documented per-cohort patient
  counts** (2D test 469 images / 76 patients; CEUS 170 images / 47
  patients). Dividing the 2D image count by the whole-dataset patient
  count understates clustering for 2D and overstates it for CEUS.
- `youden_threshold()` must be run on validation data only. Re-running it
  on test or CEUS data invalidates the prespecification.
- The segmentation threshold was **selected** on the 2D validation split
  and the selection returned 0.5. Sweeping it over the stored probability
  maps reproduces the published Dice at 0.50 to within 0.0003 and at no
  other value tested.
- Mean predictive entropy is evaluated over the union of the binarised
  member masks dilated by 10 pixels, not over the whole frame. A
  whole-frame mean is dominated by confidently empty background.

## Licence

Code released under the MIT Licence (`LICENSE`). The MMOTU dataset
carries its own licence from its original authors and is not covered by
this one.
