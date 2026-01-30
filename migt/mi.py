import cv2
import numpy as np
from skimage.metrics import normalized_mutual_information as nmi
from sklearn.metrics import mutual_info_score


# -----------------------------------
# Robust image loader (optional resize)
# -----------------------------------
def load_image(path, target_size=None):
    """
    Loads image from disk.
    If target_size is provided, image is resized to (width, height).
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")

    # Convert fake-grayscale (3 identical channels) to true grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        if (
            np.allclose(img[:, :, 0], img[:, :, 1]) and
            np.allclose(img[:, :, 1], img[:, :, 2])
        ):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if target_size is not None:
        img = cv2.resize(img, target_size)

    return img


def is_grayscale(img):
    return img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)


# -----------------------------------
# Gray MI (histogram-based, MIGT-style)
# -----------------------------------
def compute_gray_mi(img1, img2, bins=64):
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Reject constant images
    if img1.std() < 1e-6 or img2.std() < 1e-6:
        return 0.0

    hist, _, _ = np.histogram2d(
        img1.ravel(),
        img2.ravel(),
        bins=bins
    )

    return float(mutual_info_score(None, None, contingency=hist))


# -----------------------------------
# Color MI (channel-wise, normalized)
# -----------------------------------
def compute_color_mi(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    mi_vals = []
    for c in range(3):
        if img1[:, :, c].std() < 1e-6 or img2[:, :, c].std() < 1e-6:
            mi_vals.append(0.0)
        else:
            mi_vals.append(
                nmi(img1[:, :, c], img2[:, :, c])
            )

    return float(np.mean(mi_vals))


# -----------------------------------
# Unified MI interface (STRICT by default)
# -----------------------------------
def compute_mi(
    img1,
    img2,
    mode="auto",
    expected_size=None
):
    """
    Computes mutual information between two images.

    expected_size:
        None        -> strict mode (shapes must already match)
        (w, h)      -> both images must match this size
    """

    # --- Size enforcement ---
    if expected_size is not None:
        if img1.shape[:2][::-1] != expected_size or img2.shape[:2][::-1] != expected_size:
            raise ValueError(
                f"Image size mismatch: expected {expected_size}, "
                f"got {img1.shape} vs {img2.shape}"
            )
    else:
        # strict shape check
        if img1.shape[:2] != img2.shape[:2]:
            raise ValueError(
                f"Image shape mismatch: {img1.shape} vs {img2.shape}"
            )

    if mode == "grayscale":
        return compute_gray_mi(img1, img2)

    if mode == "color":
        return compute_color_mi(img1, img2)

    # auto
    if is_grayscale(img1) and is_grayscale(img2):
        return compute_gray_mi(img1, img2)
    else:
        return compute_color_mi(img1, img2)
