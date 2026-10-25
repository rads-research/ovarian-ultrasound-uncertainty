# Input file schemas

Two files drive the entire analysis and all six figures.

## `results/per_image.csv` — one row per image

| Column | Type | Required | Description |
|---|---|---|---|
| `image_id` | str | yes | Unique identifier; must match filenames in `silent_failures/` |
| `cohort` | str | yes | One of `2d_validation`, `2d_test`, `ceus` |
| `dice` | float | yes | Dice similarity coefficient, ensemble prediction vs reference |
| `disagreement` | float | yes | Ensemble disagreement (1 − mean pairwise Dice) |
| `mean_entropy` | float | no | Prespecified secondary uncertainty measure |
| `iou`, `nsd`, `hd95` | float | no | Secondary overlap and distance metrics |
| `ref_area_frac` | float | no | Reference mask area as a fraction of image area |
| `pred_area_frac` | float | no | Predicted mask area as a fraction of image area |
| `group_id` | str | no | Near-duplicate group; the bootstrap resampling unit |
| `morphology` | str | no | `regular` or `irregular` (single component, solidity ≥ 0.95) |
| `size_tertile` | str | no | `small`, `medium`, `large`; cut points from development data |

Derived in code, not stored: `failure` (`dice < 0.50`), `referred`
(`disagreement >= 0.3671`), `silent` (`failure & ~referred`).

Also derived from this file rather than stored: risk-coverage curves,
AURC, ROC and precision-recall curves. Keeping them derived means they
cannot disagree with the scatter plots.

## `results/reliability.csv` — pixel-level calibration bins

| Column | Type | Description |
|---|---|---|
| `cohort` | str | `2d_test` or `ceus` |
| `source` | str | `ensemble`, or a member name such as `member_00` |
| `region` | str | `all_pixels` or `lesion` |
| `bin_index` | int | 0 … 14 (15 equal-width probability bins) |
| `mean_pred` | float | Mean predicted probability in the bin |
| `observed_frequency` | float | Observed foreground frequency in the bin |
| `n_pixels` | int | Pixels contributing to the bin |

The lesion region is the union of the 15 binarised member predictions,
dilated by 10 iterations of a 3×3 square element at native resolution
after letterbox padding is removed; where no member predicts foreground,
the whole image is used.

## `results/silent_failures/` — example images

Three files per example, all the same pixel dimensions:

```
<image_id>_image.png    grayscale frame
<image_id>_ref.png      binary reference mask
<image_id>_pred.png     binary ensemble mask
```

Grayscale, RGB and RGBA PNGs are all accepted; masks are thresholded at
0.5. `supp_figure2()` spans the observed Dice range by default rather
than taking the worst cases, because the lowest-Dice examples are all
near zero and produce visually identical panels. Pass `image_ids=[...]`
to choose explicitly.
