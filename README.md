<p align="center">
  <img width="276" height="200" alt="logo" src="https://github.com/user-attachments/assets/96aa5c44-42ed-4ec6-b0d9-3a8a75c2d1f2" />

</p>

# MIGT — Mutual Information Guided Training

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

This implementation corresponds to the dataset partitioning stage of the MIGT framework described in :
L. Shahmiri, P. Wong and L. S. Dooley, "Accurate Medicinal Plant Identification in Natural Environments by Embedding Mutual Information in a Convolution Neural Network Model," 2022 IEEE 5th International Conference on Image Processing Applications and Systems (IPAS), Genova, Italy, 2022, pp. 1-6, doi: 10.1109/IPAS55744.2022.10053008. 

