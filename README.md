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
 └── val/        # created only if val is enabled
```
## Advanced Usage
```python
from migt import MIGTSplitter

splitter = MIGTSplitter(
    dataset_root="dataset",
    mode="auto",
    bins=4,
    min_bin=13,
    train=0.6,
    test=0.3,
    val=0.1,
    strict_shape=True,
    seed=42
)

splitter.run("migt_output")
```
## Parameters
```text
| Parameter        | Type          | Description                                          |
|------------------|---------------|------------------------------------------------------|
| dataset_root     | str           | Path to dataset directory                            |
| mode             | str           | MI mode: auto / grayscale / color                    |
| bins             | int           | Initial number of histogram bins                     |
| min_bin          | int           | Minimum samples required per bin                     |
| train            | float         | Training split ratio                                 |
| test             | float         | Test split ratio                                     |
| val              | float or None | Validation split ratio (optional)                    |
| strict_shape     | bool          | Enforce identical image sizes                        |
| resize_to        | tuple or None | Resize images for MI computation                     |
| seed             | int           | Random seed                                          |
```
## Mutual Information Modes

- auto (default): Automatically selects grayscale or color MI

- grayscale: Histogram-based MI on grayscale images

- color: Channel-wise normalized MI on RGB images

## Image Size Handling

1️⃣ Strict Shape Mode (Research-Safe)

```python
strict_shape=True
resize_to=None
```
- All images must have identical dimensions

- Mismatched images are skipped

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
 -bin merging is performed

## Fallback Rules (Deterministic)
### Case A — Small Classes (Initial size < min_bin)

- Histogram binning is skipped

- User-selected strategy is applied:

 - random

 - mi_quantile (2-quantile split)

### Case B — Large Classes, Histogram Failure

- If binning fails even at 2 bins:

 - Forced MI-based 3-quantile split

 - Random split is never used

 - User preference is ignored intentionally
## Decision Summary

```text
| Situation                               | Strategy Applied                    |
|-----------------------------------------|-------------------------------------|
| Class size < min_bin                    | User choice (random / mi_quantile)  |
| Histogram binning succeeds              | Histogram-based split               |
| Histogram fails at 2 bins               | Forced MI-based 3-quantile          |
| Random after histogram failure          | ❌ Never                            |
```
## Reference Image Handling

- First image of each class is the reference

- Reference image always goes to training set

## Train/Test Ratio Note

MI is computed on N − 1 images (excluding reference).
The reference image is added afterward, which may slightly
shift counts (e.g., 37 instead of 36).

This behavior is expected and deterministic.

## Guarantees

MIGT guarantees:

✅ No image loss

✅ No class skipping

✅ Distribution-aware splits

✅ Deterministic behavior

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

This work is based on:

L. Shahmiri, P. Wong, and L. S. Dooley,
Accurate Medicinal Plant Identification in Natural Environments by Embedding Mutual Information in a Convolution Neural Network Model,
IEEE IPAS 2022, https://ieeexplore.ieee.org/abstract/document/10053008





