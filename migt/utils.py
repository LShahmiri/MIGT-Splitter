import numpy as np


def split_counts(n, ratios):
    """
    Ensures sum(counts) == n exactly
    """
    ratios = np.array(ratios, dtype=float)
    counts = np.floor(ratios * n).astype(int)

    remainder = n - counts.sum()

    counts[0] += remainder

    return counts.tolist()
