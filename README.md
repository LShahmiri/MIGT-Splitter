<p align="center">
  <img width="276" height="200" alt="logo" src="https://github.com/user-attachments/assets/96aa5c44-42ed-4ec6-b0d9-3a8a75c2d1f2" />
</p>

<h1 align="center">MIGT: Mutual Information Guided Training</h1>

<p align="center">
  <b>Model-agnostic dataset partitioning using Mutual Information</b>
</p>

---

## Overview

**MIGT (Mutual Information Guided Training)** is a **model-agnostic dataset partitioning framework**
that splits image datasets into **train / test / (optional) validation** sets using
**Mutual Information (MI)** instead of random sampling.

The main objective of MIGT is to **preserve the information distribution**
of samples across dataset splits and reduce dataset bias.

Unlike random splitting, MIGT:
- Preserves easy and hard samples proportionally
- Maintains feature similarity across splits
- Reduces distributional skew
- Improves generalization stability

MIGT is compatible with **CNNs, Vision Transformers (ViTs)**, and any vision-based model.

---

## Key Features

- ✅ Mutual Information–guided dataset partitioning
- ✅ Excel-style **distribution-aware histogram binning**
- ✅ Adaptive bin reduction (**4 → 3 → 2**)
- ✅ Deterministic fallback strategies
- ✅ Optional validation split
- ✅ Research-safe strict shape mode
- ✅ Optional practical resizing mode
- ✅ Model-agnostic and framework-independent
- ✅ No data loss, no class skipping
- ✅ No data leakage: no image in two splits
- ✅ No split left empty by rounding
- ✅ Every image reported in `migt_manifest.csv`

**What's new in 1.2.0:** defaults as in the paper (`min_bin=10`, small classes
split at random), no empty test/validation split, no image lost without a
warning, no reference image copied into two splits, reproducible order. See
[CHANGELOG.md](CHANGELOG.md).

---

## Installation

```bash
pip install migt
```
## Dataset Structure
```text
dataset/
 ├── class1/
 │    ├── img1.jpg
 │    ├── img2.jpg
 ├── class2/
 │    ├── img1.jpg
 │    ├── img2.jpg
```
## Basic Usage
```python
from migt import MIGTSplitter

splitter = MIGTSplitter(
    dataset_root="path/to/dataset"
)

splitter.run(output_root="migt_output")
```
## Output Structure
```text
migt_output/
 ├── train/
 ├── test/
 ├── val/                 # created only if val is enabled
 └── migt_manifest.csv    # every image: class, split, role, MI, bin
```
`splitter.plan()` computes the same partition without copying files.
## Advanced Usage
```python
from migt import MIGTSplitter

splitter = MIGTSplitter(
    dataset_root="dataset",
    mode="auto",
    bins=4,
    min_bin=10,
    train=0.6,
    test=0.3,
    val=0.1,
    strict_shape=True,
    seed=42
)

splitter.run("migt_output")
```
## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset_root` | str | – | Path to dataset directory |
| `mode` | str | `"auto"` | MI mode: `auto` / `grayscale` / `color` |
| `bins` | int | `4` | Initial number of histogram bins |
| `min_bin` | int | `10` | Minimum samples required per bin (value of the paper) |
| `train` | float | `0.8` | Training split ratio |
| `test` | float | `0.2` | Test split ratio |
| `val` | float or None | `None` | Validation split ratio (optional) |
| `strict_shape` | bool | `True` | Enforce identical image sizes |
| `resize_to` | tuple or None | `None` | Resize images for MI computation (required if `strict_shape=False`) |
| `seed` | int | `42` | Random seed |
| `verbose` | bool | `True` | Print progress |
| `low_sample_strategy` | str | `"random"` | Classes with <= `min_bin` images: `random` (as in the paper) / `mi_quantile` |
| `rounding` | str | `"class"` | `class`: split sizes rounded per class, no empty split; `per_bin`: every bin rounded separately (up to 1.1.1) |
| `unscored` | str | `"include"` | Images without MI score: `include` (split at random, with a warning) / `skip` (left out, with a warning) |
## Mutual Information Modes

- auto (default): grayscale MI if the reference image is grayscale, otherwise colour MI
  (a black-and-white photo in a colour class is replicated to three channels)

- grayscale: Histogram-based MI on grayscale images

- color: Channel-wise normalized MI on RGB images

## Image Size Handling

1️⃣ Strict Shape Mode (Research-Safe)

```python
strict_shape=True
resize_to=None
```
- All images must have identical dimensions

- Mismatched images get no MI score: they are split at random as an extra
  stratum of their class and reported with a warning (`unscored="include"`),
  or left out with a warning (`unscored="skip"`)

- No interpolation applied

- Recommended for scientific experiments

2️⃣ Practical Mode (Optional Resizing)

```python
strict_shape=False
resize_to=(224, 224)
```
- Images resized in memory for MI computation

- Supports mixed-resolution datasets

- Original images are copied unchanged

## Histogram Binning Strategy (Final)

For each class:

1- Compute MI relative to a reference image

2- Construct value-based histogram bins

3- Start with user-defined bins (default: 4)

 If any bin has fewer than min_bin samples:

 - Reduce bins sequentially: 4 → 3 → 2

 - Recompute histogram from scratch each time

 - No bins are merged; empty bins are dropped before the `min_bin` test

## Fallback Rules (Deterministic)
### Case A — Small Classes (size <= min_bin)

- Histogram binning is skipped

- User-selected strategy is applied:

 - random (default, as in the paper): all images split at random by the ratios

 - mi_quantile: the reference goes to train and the two MI halves are used as strata

### Case B — Large Classes, Histogram Failure

- If binning fails even at 2 bins:

 - Forced MI-based 3-quantile split

 - Random split is never used

 - User preference is ignored intentionally
## Decision Summary

| Situation | Strategy Applied |
|---|---|
| Class size <= `min_bin` | `random` (default) / `mi_quantile` |
| Histogram binning succeeds | Histogram-based split |
| Histogram fails at 2 bins | Forced MI-based 3-quantile |
| Random after histogram failure | ❌ Never |
## Reference Image Handling

- First image of each class is the reference

- Reference image always goes to training set (MI-guided classes), and only there

## Train/Test Ratio Note

The train/test/val sizes are rounded once per class (largest remainder, at
least one image per split when the class has at least as many images as
splits), including the reference image, and then spread over the MI bins in
proportion to their sizes. Example with 70/20/10: a 15-image class split into
MI parts of 4, 5 and 5 images gets 11/3/1 images. With `rounding="per_bin"`
(behaviour up to 1.1.1) every bin is rounded separately and the remainder goes
to train, which gave 13/2/0 for the same class.

## Guarantees

MIGT guarantees:

✅ No image loss: every image is in exactly one split; images without an MI score are reported

✅ No data leakage: no image is copied into two splits; a non-empty output folder is refused

✅ No class skipping

✅ No split left empty by rounding (`rounding="class"`)

✅ Distribution-aware splits

✅ Deterministic behavior (sorted classes and files, private seeded random generator)

✅ Randomness only when statistically justified

## Validation Split (Optional)
```python
MIGTSplitter(
    train=0.8,
    test=0.2,
    val=None
)
```
- Only train/ and test/ folders are created

- No validation directory is generated

With `val`, for example `train=0.7, test=0.2, val=0.1`, a val/ folder is
created as well.

## MIGT in the GMM-MIGT package

The same MIGT partition is also available in
[gmm-migt](https://github.com/LShahmiri/GMM-MIGT) with
`GMMMIGTSplitter(dataset_root, method="migt")`; with the same parameters both
packages give identical splits.

## Testing

```bash
git clone https://github.com/LShahmiri/MIGT-Splitter.git
cd MIGT-Splitter
pip install -e ".[test]"
pytest
```

## Dependencies

```text
numpy>=1.23
scipy>=1.8
scikit-learn>=1.2
scikit-image>=0.19
opencv-python>=4.5
pillow>=9.0
tqdm>=4.64
```
## Reference

This work is based on the following paper; if you use MIGT, please cite it:

L. Shahmiri, P. Wong, and L. S. Dooley,
Accurate Medicinal Plant Identification in Natural Environments by Embedding Mutual Information in a Convolution Neural Network Model,
IEEE IPAS 2022, https://ieeexplore.ieee.org/abstract/document/10053008

## License

MIT License. Copyright (c) 2026 Lida Shahmiri. See [LICENSE](LICENSE).
