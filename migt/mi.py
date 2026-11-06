import cv2
import numpy as np
from skimage.metrics import normalized_mutual_information as nmi
from sklearn.metrics import mutual_info_score

GRAY_MI_BINS = 64


# -----------------------------------
# Robust image loader (optional resize)
# -----------------------------------
def load_image(path, target_size=None):
    """
    Loads image from disk (OpenCV BGR order).
    - Paths with non-ASCII characters are supported.
    - An alpha channel is discarded.
    - Three identical channels ("fake grayscale") become a true grayscale image.
    - If target_size=(width, height) is given, the image is resized in memory.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Cannot load image: {path} ({exc})") from exc
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if img is None:
        raise ValueError(f"Cannot load image: {path}")

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    # Convert fake-grayscale (3 identical channels) to true grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        if np.allclose(img[:, :, 0], img[:, :, 1]) and np.allclose(img[:, :, 1], img[:, :, 2]):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if target_size is not None:
        img = cv2.resize(img, tuple(int(v) for v in target_size))

    return img


def is_grayscale(img):
    return img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)


def _as_cv_supported(img):
    if img.dtype in (np.uint8, np.uint16, np.float32):
        return img
    return img.astype(np.float32)


# -----------------------------------
# Gray MI (histogram-based, MIGT-style)
# -----------------------------------
def compute_gray_mi(img1, img2, bins=GRAY_MI_BINS):
    if img1.ndim == 3:
        img1 = cv2.cvtColor(_as_cv_supported(img1), cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(_as_cv_supported(img2), cv2.COLOR_BGR2GRAY)

    # Reject constant images
    if img1.std() < 1e-6 or img2.std() < 1e-6:
        return 0.0

    hist, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    return float(mutual_info_score(None, None, contingency=hist))


# -----------------------------------
# Color MI (channel-wise, normalized)
# -----------------------------------
def compute_color_mi(img1, img2):
    if img1.ndim != 3 or img1.shape[2] != 3 or img2.ndim != 3 or img2.shape[2] != 3:
        raise ValueError(
            f"Color MI requires two 3-channel images (got {img1.shape} and {img2.shape}). "
            "Use mode='grayscale' or mode='auto' for grayscale data."
        )
    img1 = cv2.cvtColor(_as_cv_supported(img1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(_as_cv_supported(img2), cv2.COLOR_BGR2RGB)

    mi_vals = []
    for c in range(3):
        if img1[:, :, c].std() < 1e-6 or img2[:, :, c].std() < 1e-6:
            mi_vals.append(0.0)
        else:
            mi_vals.append(nmi(img1[:, :, c], img2[:, :, c]))

    return float(np.mean(mi_vals))


# -----------------------------------
# Unified MI interface (STRICT by default)
# -----------------------------------
def compute_mi(img1, img2, mode="auto", expected_size=None):
    """
    Computes mutual information between two images; img1 is the reference.

    mode:
        "auto"      -> grayscale MI if the reference is grayscale, otherwise
                       colour MI (a grayscale img2 is replicated to three
                       channels, so all scores of a class are on one scale)
        "grayscale" -> always grayscale MI
        "color"     -> always channel-wise colour MI
    expected_size:
        None        -> strict mode (shapes must already match)
        (w, h)      -> both images must match this size
    """
    if expected_size is not None:
        if img1.shape[:2][::-1] != tuple(expected_size) or img2.shape[:2][::-1] != tuple(expected_size):
            raise ValueError(
                f"Image size mismatch: expected {tuple(expected_size)}, got {img1.shape} vs {img2.shape}"
            )
    elif img1.shape[:2] != img2.shape[:2]:
        raise ValueError(f"Image shape mismatch: {img1.shape} vs {img2.shape}")

    if mode == "grayscale":
        return compute_gray_mi(img1, img2)
    if mode == "auto" and is_grayscale(img1):
        return compute_gray_mi(img1, img2)
    if is_grayscale(img2) and not is_grayscale(img1):
        img2 = np.repeat(img2.reshape(img2.shape[0], img2.shape[1], 1), 3, axis=2)
    return compute_color_mi(img1, img2)
