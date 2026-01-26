"""
Basic usage example for MIGT (Mutual Information Guided Training)
"""

from migt import MIGTSplitter

# Dataset structure:
# dataset/
# ├── class_1/
# ├── class_2/
# └── class_3/

splitter = MIGTSplitter(
    dataset_root="dataset"
)

splitter.run(output_root="migt_output")

print("MIGT dataset splitting completed.")
