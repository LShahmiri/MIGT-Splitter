import cv2
import numpy as np
from skimage.metrics import normalized_mutual_information as nmi
from sklearn.metrics import mutual_info_score


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return img


def is_grayscale(img):
    return len(img.shape) == 2 or img.shape[2] == 1


def compute_gray_mi(img1, img2, bins=64):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    hist, _, _ = np.histogram2d(
        img1.ravel(), img2.ravel(), bins=bins
    )
    return mutual_info_score(None, None, contingency=hist)


def compute_color_mi(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    mi_vals = []
    for c in range(3):
        mi_vals.append(nmi(img1[:, :, c], img2[:, :, c]))
    return float(np.mean(mi_vals))


def compute_mi(img1, img2, mode="auto"):
    if mode == "grayscale":
        return compute_gray_mi(img1, img2)

    if mode == "color":
        return compute_color_mi(img1, img2)

    # auto
    if is_grayscale(img1):
        return compute_gray_mi(img1, img2)
    else:
        return compute_color_mi(img1, img2)
