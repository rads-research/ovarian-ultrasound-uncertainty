# Data

The MMOTU multi-modality ovarian tumour ultrasound dataset is
distributed by its original authors and is **not** redistributed here.
Obtain it from the authors' public release and arrange it as:

```
data/MMOTU/
├── OTU_2d/
│   ├── images/
│   └── annotations/
└── OTU_CEUS/
    ├── images/
    └── annotations/
```

## Composition

| Subset | Images | Patients |
|---|---|---|
| 2D, development split | 1,000 | 171 |
| 2D, held-out test split | 469 | 76 |
| CEUS, both splits (all used here) | 170 | 47 |
| **Total** | **1,639** | **294** |

Two points that matter for the analysis:

1. **294 is the whole-dataset patient count**, covering both modalities.
   The 2D subset alone is 247 patients. Dividing 1,469 by 294 gives a
   cluster size of 5.0, which is wrong for either cohort; the code uses
   the per-cohort counts above.
2. **Per-image patient identifiers are not released.** Only the
   per-split counts are. Near-duplicate images are therefore grouped by
   normalised cross-correlation of downsampled images and those groups
   are used as the resampling unit, which recovers little of the true
   patient structure. The design-effect analysis quantifies the likely
   cost of that.

The 2D subset contains 216 colour Doppler flow images whose per-image
modality labels are not released, so the subset is retained in full and
described as conventional 2D ultrasound throughout.
