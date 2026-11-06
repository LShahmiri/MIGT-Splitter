import os

import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
ROUNDING_MODES = ("class", "per_bin")


def split_counts(n, ratios):
    """
    Per-split counts that sum to n exactly: floor(ratio * n), remainder to the
    first split (train). Used with rounding="per_bin".
    ratio * n is rounded to 9 decimals first so that floating-point error
    (e.g. 0.29 * 100 = 28.999999999999996) cannot move an item.
    """
    ratios = np.asarray(ratios, dtype=float)
    counts = np.floor(np.round(ratios * n, 9)).astype(int)
    counts[0] += n - int(counts.sum())
    return counts.tolist()


def class_counts(n, ratios, min_one=True):
    """
    Per-split counts of a class of n images, rounded once for the whole class
    (largest remainder; ties go to the larger ratio, then the earlier split).
    With min_one=True and n >= number of splits, every split gets at least one
    image, taken from the largest split. Example: n=4, 0.7/0.2/0.1 -> 2/1/1.
    """
    r = np.asarray(ratios, dtype=float)
    exact = np.round(r * n, 9)
    counts = np.floor(exact).astype(int)
    rest = n - int(counts.sum())
    order = sorted(range(len(r)), key=lambda i: (-(exact[i] - counts[i]), -r[i], i))
    for i in order[:rest]:
        counts[i] += 1
    if min_one and n >= len(r):
        for i in range(len(r)):
            if counts[i] == 0:
                counts[int(np.argmax(counts))] -= 1
                counts[i] = 1
    return counts.tolist()


def _capped_largest_remainder(total, weights, caps):
    caps = [int(c) for c in caps]
    w = np.asarray(weights, dtype=float)
    if total <= 0 or w.sum() <= 0:
        return [0] * len(caps)
    exact = np.round(total * w / w.sum(), 9)
    out = [min(int(np.floor(e)), c) for e, c in zip(exact, caps)]
    while sum(out) < total:
        free = [i for i in range(len(out)) if out[i] < caps[i]]
        if not free:
            raise ValueError("total exceeds the sum of caps")
        i = max(free, key=lambda k: (exact[k] - out[k], caps[k] - out[k], -k))
        out[i] += 1
    return out


def stratified_counts(sizes, ratios, n_fixed_train=0, min_one=True):
    """
    Per-stratum split counts with the split sizes fixed for the whole class.
    The class total N = sum(sizes) + n_fixed_train (the reference image, already
    in train) is rounded once with class_counts; the test/val counts are then
    spread over the strata in proportion to their sizes and the rest of every
    stratum goes to train. Returns [[train, test(, val)], ...].
    """
    sizes = [int(s) for s in sizes]
    n = sum(sizes) + int(n_fixed_train)
    q = class_counts(n, ratios, min_one)
    if q[0] < n_fixed_train:
        for i in range(len(q) - 1, 0, -1):
            while q[0] < n_fixed_train and q[i] > 0:
                q[i] -= 1
                q[0] += 1
    left = sizes[:]
    out = [[0] * len(q) for _ in sizes]
    for s in range(1, len(q)):
        part = _capped_largest_remainder(q[s], sizes, left)
        for b, k in enumerate(part):
            out[b][s] = k
            left[b] -= k
    for b in range(len(sizes)):
        out[b][0] = left[b]
    return out


def list_images(directory):
    """Sorted full paths of the image files directly inside directory."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(directory, f))
    )


def list_classes(dataset_root):
    """Sorted names of the class sub-directories of dataset_root."""
    return sorted(d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d)))
