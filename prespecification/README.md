# Prespecification

This directory holds the locked analysis plan and its SHA-256 hash.

## Important

`prespecification.yaml` must be **your original locked file, byte for
byte**. Do not regenerate, reformat or re-save it: the hash quoted in the
manuscript refers to the original, and any change — even whitespace —
produces a different hash and destroys the audit trail the lock exists to
provide.

If the file in this repository is the template rather than the original,
replace it before release and regenerate `prespecification.sha256`:

```bash
python verify_hash.py --write
```

Then confirm the printed hash matches the one in the manuscript. If it
does not, the file here is not the one that was locked.

## Checking the lock

```bash
python verify_hash.py
```

`scripts/predict.py --cohort ceus` and `scripts/run_analysis.py` call
this check and refuse to load CEUS data if it fails.

## What the lock covers

Ensemble membership and combination rule; segmentation threshold;
definitions of major failure and silent failure; uncertainty measures;
the referral threshold and how it was chosen; subgroup definitions; and
the statistical analysis plan including the six hypotheses and the
multiplicity adjustment.

Four analyses were added **after** the locked results were examined and
are outside the lock: the equivalent-diameter error, the comparison of
uncertainty measures, the ensemble-size and composition ablations, and
calibration. They are labelled exploratory throughout and are not
corrected for multiplicity.
