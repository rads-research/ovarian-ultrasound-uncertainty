"""Shared configuration: paths, prespecified constants, and figure styling.

Every value under PRESPECIFIED was fixed before any contrast-enhanced
ultrasound image was read and must not be altered when re-running the
analysis. Values marked FILL_IN are environment-specific and must be
completed before the repository is released.
"""

from __future__ import annotations

import warnings
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
PRESPEC_DIR = ROOT / "prespecification"

for _d in (RESULTS_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Prespecified analysis constants  (DO NOT EDIT)
# --------------------------------------------------------------------------

SEEDS = (1337, 2337, 3337, 4337, 5337)
ARCHITECTURES = ("unet_resnet34", "nnunet2d", "segformer_b2")

IMAGE_SIZE = 384                 # square, aspect-ratio-preserving pad
SEGMENTATION_THRESHOLD = 0.50    # probability -> binary mask
MAJOR_FAILURE_DICE = 0.50        # Dice < 0.50 == major failure
REFERRAL_THRESHOLD = 0.3671      # Youden index on 2D validation only

NSD_TOLERANCE_PX = 3
CALIBRATION_BINS = 15
LESION_ROI_DILATION_ITERS = 10   # 3x3 square element, native resolution

N_BOOTSTRAP = 2000               # primary analyses
N_BOOTSTRAP_SECONDARY = 500      # individual members, metric comparison
ALPHA = 0.05

# Alternative major-failure definitions (prespecified sensitivity analysis)
ALT_FAILURE_THRESHOLDS = (0.40, 0.50, 0.60, 0.70)

# Documented MMOTU cohort sizes, used for the design-effect analysis.
# Source: MMOTU dataset paper, Table 1 (per-split patient counts).
#   2D   : 1000 train / 171 patients + 469 test / 76 patients
#   CEUS :   70 train /  20 patients + 100 test / 27 patients
COHORT_PATIENTS = {
    "2d_validation": None,       # subset of the 171 development patients
    "2d_test": 76,
    "ceus": 47,                  # 20 + 27, the full CEUS subset is used
}
COHORT_IMAGES = {"2d_validation": 204, "2d_test": 469, "ceus": 170}
DESIGN_EFFECT_ICCS = (0.3, 0.6)

COHORT_LABELS = {
    "2d_validation": "2D validation",
    "2d_test": "2D internal test",
    "ceus": "CEUS",
}
COHORT_ORDER = ("2d_validation", "2d_test", "ceus")

# --------------------------------------------------------------------------
# Training hyperparameters  (FILL_IN before release)
# --------------------------------------------------------------------------


class _FillIn:
    """Sentinel that fails loudly rather than silently publishing a guess."""

    def __init__(self, what: str) -> None:
        self.what = what

    def __repr__(self) -> str:
        return f"<FILL_IN: {self.what}>"

    def __bool__(self):
        raise ValueError(
            f"'{self.what}' has not been set in src/config.py. "
            "Enter the value actually used for the published runs."
        )


FILL_IN = _FillIn

TRAINING = {
    "optimiser": "AdamW",                       # all three architectures
    # Per architecture, not a single value. These are read from the training
    # notebook source; the prespecification lock does not record them, so
    # they cannot be cross-checked against the run's own artefacts.
    "learning_rate": {"unet_resnet34": 3e-4, "nnunet2d": 1e-3,
                      "segformer_b2": 6e-5},
    "weight_decay": {"unet_resnet34": 1e-4, "nnunet2d": 1e-5,
                     "segformer_b2": 1e-2},
    "lr_schedule": "3-epoch linear warmup, then cosine annealing",
    "batch_size": 8,
    "batch_size_min": 2,                        # see note below
    "batch_size_auto_reduce": True,
    "max_epochs": 80,
    "loss": "Dice + cross-entropy (equal weight)",
    "model_selection": "epoch of highest validation Dice",
    "normalisation": "ImageNet mean/std",
}

# Batch size was reduced automatically when GPU memory was insufficient,
# so members may not all have trained at 8. Record the size actually used
# per member here if known: it is the one uncontrolled difference between
# members besides architecture and seed, and it bears on the exploratory
# architecture-versus-seed comparison.
BATCH_SIZE_PER_MEMBER: dict[str, int] = {}      # e.g. {"segformer_b2__seed1337": 4}

AUGMENTATION = {
    # transform: (parameters, probability of application)
    "horizontal_flip": (None, 0.50),
    "translation": ("+/-6.25% of image size", 0.50),
    "scaling": ("0.85-1.15x", 0.50),
    "rotation": ("+/-15 degrees", 0.50),
    "brightness_contrast": ("+/-20% each", 0.50),
    "gaussian_noise": (FILL_IN("variance range used for the PUBLISHED run"), 0.20),
    "gaussian_blur": ("kernel size 3-5", 0.15),
    "elastic_deformation": ("alpha=30, sigma=6", 0.15),
    # No vertical flip was applied.
}

# The Gaussian-noise setting differed between notebook versions. Record the
# value from the single run that produced the published results, and pin that
# version below; a repository that cannot say which run it describes does not
# support the reproducibility claim it is there to make.
# The run is identified by RUN_ID in GCS, not by which notebook file was open.
# RUN_ID is recorded by the run itself and is authoritative; the notebook
# filename is incidental and was not confirmed, so it is not asserted here.
RUN_ID = "run_full_v3"
# Object-store location of the run artefacts. Set to your own path; the
# original is withheld here because the repository is anonymised.
GCS_RUN_ROOT = "[OBJECT STORE PATH]"
# The run is identified by RUN_ID and by the completion timestamps its own
# stages recorded; the notebook filename was not recoverable and is not
# asserted. Stages ran 2026-08-31 00:34 to 08:36 UTC, training occupying
# 27,160 s of that.
SOURCE_NOTEBOOK = None
STAGE_COMPLETION_UTC = {
    "02_splits": "2026-08-31T00:34:30", "03_train": "2026-08-31T08:07:25",
    "04_predict_internal": "2026-08-31T08:23:31", "05_lock": "2026-08-31T08:25:21",
    "06_internal_eval": "2026-08-31T08:28:26", "07_ceus_stress_test": "2026-08-31T08:34:43",
    "08_analysis": "2026-08-31T08:35:36", "09_figures": "2026-08-31T08:36:55",
    "10_report": "2026-08-31T08:36:57",
}

ENVIRONMENT = {
    "python": "3.13.16",
    "pytorch": "2.11.0+cu130",
    "cuda": "13.0",
    "gpu": "NVIDIA A100",                       # add memory variant if known
    "n_permutations": 2000,                     # N_PERMUTATION, run config
}

# Implementation details of run_full_v3, taken from the prespecification
# lock. These are not results; they are what the code did, and they are
# required to reproduce the reported numbers. Analyses that the manuscript
# does not report are deliberately absent from this file.
RUN_SETTINGS = {
    "img_size": 384,
    "amp": True,
    "grad_clip": 12.0,
    "early_stop_patience": 25,                  # so a member may stop before 80
    "warmup_epochs": 3,
    "deterministic": False,                     # ~20-30% faster; rerun is not
                                                # bitwise reproducible
    "num_workers": 2,
    "bootstrap_seed": 777,
    "split_seed": 20240101,
    "dev_val_fraction": 0.20,                   # by pseudo-patient
    # Resolved against the prespecification lock and confirmed empirically:
    # the threshold was SELECTED on 2D validation data (mode "dev_optimal")
    # and the selection returned 0.5. Sweeping the threshold over the stored
    # probability maps reproduces the published Dice at 0.50 to within
    # 0.0003, and at no other value tested.
    "prob_threshold_mode": "dev_optimal, derived on 2D validation; value 0.5",
    "pseudo_patient_method": "perceptual hash, 64-bit, merge at Hamming <= 10",
    # From the prespecification lock:
    "test_time_augmentation": False,
    "loss": "soft Dice + BCE, deep supervision on the nnU-Net-style arm",
    "channels": "grayscale replicated to 3",
    "intensity_scaling": "divide by 255 only; no per-image or per-cohort z-scoring",
    "metric_resolution": "native image resolution; predictions inverse-letterboxed",
    "empty_mask_convention": "Dice = 1 when prediction and reference are both empty",
    "member_selection": "highest 2D validation Dice epoch, per member",
    "ensemble_combination": "unweighted mean of member sigmoid probabilities",
}

# --------------------------------------------------------------------------
# Figure styling
# --------------------------------------------------------------------------

# Okabe-Ito colour-vision-safe palette. Colour never carries information on
# its own in any figure: every series also differs by marker and line style.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#999999",
}

COHORT_STYLE = {
    "2d_validation": {"color": OKABE_ITO["grey"], "marker": "o", "ls": ":"},
    "2d_test": {"color": OKABE_ITO["blue"], "marker": "s", "ls": "-"},
    "ceus": {"color": OKABE_ITO["reddish_purple"], "marker": "^", "ls": "--"},
}

# --------------------------------------------------------------------------
# Values as printed in the manuscript.
#
# make_figures.py recomputes each of these from the per-image data and
# ABORTS if any disagrees at the precision the manuscript reports. A figure
# that contradicts the text therefore cannot be produced.
#
# AURC note: the run's own report gives AURC to 4 dp (failure: 0.0112 on 2D,
# 0.0429 on CEUS). Recomputing it reproduces those to within 1 unit in the
# 4th decimal, the residual depending on the coverage grid used. Every
# plausible convention agrees at the 3 dp the manuscript reports, so the
# reported values are convention-independent. The standard definition is
# used here: the mean of the risk-coverage curve.
# --------------------------------------------------------------------------

MANUSCRIPT_VALUES = {
    # (cohort, quantity): (value as printed, decimal places printed)
    ("2d_test", "auroc"):  (0.863, 3),
    ("2d_test", "auprc"):  (0.294, 3),
    ("ceus", "auroc"):     (0.842, 3),
    ("ceus", "auprc"):     (0.561, 3),
    ("2d_test", "aurc"):   (0.011, 3),
    ("2d_test", "aurc_random"): (0.047, 3),
    ("ceus", "aurc"):      (0.043, 3),
    ("ceus", "aurc_random"): (0.159, 3),
    ("2d_test", "mean_dice"): (0.864, 3),
    ("ceus", "mean_dice"):    (0.722, 3),
    ("2d_validation", "mean_dice"): (0.875, 3),
}

# Counts printed in Table 1 and Table 3: n, major failures, referred, silent.
MANUSCRIPT_COUNTS = {
    "2d_validation": (204, 7, 15, 1),
    "2d_test": (469, 22, 33, 13),
    "ceus": (170, 27, 47, 9),
}

# Elsevier column widths (mm -> inches)
WIDTH_SINGLE = 90 / 25.4
WIDTH_ONEHALF = 140 / 25.4
WIDTH_DOUBLE = 190 / 25.4

DPI = 600                 # combination line/halftone artwork
FONT_CANDIDATES = ("Times New Roman", "Liberation Serif", "Nimbus Roman",
                   "FreeSerif", "DejaVu Serif")
BASE_FONT_PT = 8
OUTPUT_FORMATS = ("tiff", "pdf")


def apply_style() -> str:
    """Apply the manuscript figure style. Returns the serif font chosen."""
    import matplotlib
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in FONT_CANDIDATES if f in available), None)
    if font is None:
        warnings.warn(
            "No Times-like serif font found; falling back to the matplotlib "
            "default. Install Times New Roman or Liberation Serif for output "
            "matching the manuscript."
        )
        font = matplotlib.rcParams["font.serif"][0]
    elif font != "Times New Roman":
        warnings.warn(
            f"Times New Roman not installed; using metric-compatible '{font}'."
        )

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": [font] + [f for f in FONT_CANDIDATES if f != font],
        "mathtext.fontset": "dejavuserif",
        "font.size": BASE_FONT_PT,
        "axes.titlesize": BASE_FONT_PT + 1,
        "axes.labelsize": BASE_FONT_PT,
        "xtick.labelsize": BASE_FONT_PT - 1,
        "ytick.labelsize": BASE_FONT_PT - 1,
        "legend.fontsize": BASE_FONT_PT - 1,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.0,
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,      # embed TrueType, keep text selectable
        "ps.fonttype": 42,
    })
    return font


def save_figure(fig, stem: str, formats=OUTPUT_FORMATS) -> list[Path]:
    """Save a figure to FIGURE_DIR in each requested format."""
    written = []
    for fmt in formats:
        path = FIGURE_DIR / f"{stem}.{fmt}"
        kwargs = {}
        if fmt == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, format=fmt, **kwargs)
        written.append(path)
    return written
