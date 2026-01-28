<p align="center">
  <img width="276" height="200" alt="logo" src="https://github.com/user-attachments/assets/96aa5c44-42ed-4ec6-b0d9-3a8a75c2d1f2" />


</p>
<h1 align="center">MIGT: Mutual Information Guided Training</h1>

<p align="center">
  <b>Model-agnostic dataset partitioning using Mutual Information</b>
</p>


MIGT is a model-agnostic dataset partitioning method based on Mutual Information (MI).
It is designed to reduce dataset bias by guiding train/test/validation splits
using information-theoretic similarity rather than random sampling.

## Key Features
- MI-guided dataset partitioning
- Resolution-independent (works with any image size)
- Automatic bin selection (starting from 4 bins)
- Safe defaults for train / test / validation splits
- Model-agnostic (CNN, ViT, etc.)

## Installation

```bash

pip install migt
from migt import MIGTSplitter

splitter = MIGTSplitter(
    dataset_root="path/to/dataset"
)

splitter.run(output_root="migt_output")

```
### Advanced Usage

MIGT provides flexible configuration options for controlling dataset partitioning:

```python
from migt import MIGTSplitter

splitter = MIGTSplitter(
    dataset_root="path/to/dataset",
    mode="auto",        # MI computation mode
    bins=4,             # initial number of histogram bins
    min_bin=10,         # minimum samples per bin
    train=0.6,          # training split ratio
    test=0.3,           # testing split ratio
    val=0.1,            # validation split ratio
    seed=42             # random seed for reproducibility
)

splitter.run(output_root="migt_output")


```
#### Parameters

- `dataset_root` (str): Path to the dataset directory (class-wise folders).
- `mode` (str): Mutual Information computation mode.
  - `"grayscale"`: Computes MI on grayscale images using histogram-based estimation.
  - `"color"`: Computes channel-wise normalized MI on RGB images.
  - `"auto"` (default): Automatically selects the appropriate MI formulation based on image type.
- `bins` (int): Initial number of histogram bins used for MI-guided grouping.
- `min_bin` (int): Minimum number of samples required per bin.
- `train` (float): Proportion of samples assigned to the training set.
- `test` (float): Proportion of samples assigned to the test set.
- `val` (float): Proportion of samples assigned to the validation set.
- `seed` (int): Random seed for reproducibility.

### Dataset Structure 

```text
dataset/
 ├── class1/
 │    ├── img1.jpg
 │    ├── img2.jpg
 ├── class2/
 │    ├── img1.jpg
 │    ├── img2.jpg
```

### Reference

This work is based on the MIGT approach described in:  
[L. Shahmiri, P. Wong, and L. S. Dooley, *Accurate Medicinal Plant Identification in Natural Environments by Embedding Mutual Information in a Convolution Neural Network Model*, IEEE IPAS 2022](https://ieeexplore.ieee.org/abstract/document/10053008)
