import os
import shutil
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from .mi import load_image, compute_mi
from .utils import split_counts


# --------------------------------------------------
# Bin structure
# --------------------------------------------------
@dataclass
class BinInfo:
    mi_min: float
    mi_max: float
    items: List[Tuple[str, float]]  # (path, mi)


# --------------------------------------------------
# MIGT Splitter 
# --------------------------------------------------
class MIGTSplitter:

    def __init__(
        self,
        dataset_root: str,
        mode: str = "auto",
        bins: int = 4,
        min_bin: int = 13,
        train: float = 0.8,
        test: float = 0.2,
        val: Optional[float] = None,
        strict_shape: bool = True,
        resize_to: Optional[Tuple[int, int]] = None,
        seed: int = 42,
        verbose: bool = True,
        low_sample_strategy: str = "mi_quantile",  # "mi_quantile" | "random"
    ):
        self.dataset_root = dataset_root
        self.mode = mode
        self.start_bins = int(bins)
        self.min_bin = int(min_bin)
        self.strict_shape = strict_shape
        self.resize_to = resize_to
        self.seed = seed
        self.verbose = verbose
        self.low_sample_strategy = low_sample_strategy

        if not strict_shape and resize_to is None:
            raise ValueError("When strict_shape=False, resize_to must be provided, e.g. resize_to=(224,224).")

        if low_sample_strategy not in ("mi_quantile", "random"):
            raise ValueError("low_sample_strategy must be 'mi_quantile' or 'random'.")

        # ratios
        if val is None:
            self.val = None
            self.ratios = [float(train), float(test)]
        else:
            self.val = float(val)
            self.ratios = [float(train), float(test), self.val]

        if any(r <= 0 for r in self.ratios):
            raise ValueError("train/test/val must be > 0.")
        s = sum(self.ratios)
        if not np.isclose(s, 1.0):
            raise ValueError(f"train/test/val ratios must sum to 1.0 (got {s}).")

        random.seed(seed)
        np.random.seed(seed)

    # --------------------------------------------------
    # Utils
    # --------------------------------------------------
    def _ensure_dirs(self, out, cls):
        os.makedirs(os.path.join(out, "train", cls), exist_ok=True)
        os.makedirs(os.path.join(out, "test", cls), exist_ok=True)
        if self.val is not None:
            os.makedirs(os.path.join(out, "val", cls), exist_ok=True)

    def _list_images(self, d):
        return sorted([
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))
        ])

    # --------------------------------------------------
    # Split copy (no loss)
    # --------------------------------------------------
    def _copy_split(self, paths, out, cls):
        """
        Copy ALL paths with no loss:
        - if val is None: leftovers go to test
        - if val exists: leftovers go to val
        """
        counts = split_counts(len(paths), self.ratios)

        n_train = counts[0]
        n_test = counts[1]

        train_paths = paths[:n_train]
        test_paths = paths[n_train:n_train + n_test]
        rest = paths[n_train + n_test:]

        for p in train_paths:
            shutil.copy(p, os.path.join(out, "train", cls))
        for p in test_paths:
            shutil.copy(p, os.path.join(out, "test", cls))

        if self.val is not None:
            for p in rest:
                shutil.copy(p, os.path.join(out, "val", cls))
        else:
            for p in rest:
                shutil.copy(p, os.path.join(out, "test", cls))

    # --------------------------------------------------
    # TRUE histogram binning (value-based)
    # --------------------------------------------------
    def _histogram_bins(self, mi_list: List[Tuple[str, float]], num_bins: int) -> List[BinInfo]:
        values = np.array([mi for _, mi in mi_list], dtype=float)
        lo, hi = float(values.min()), float(values.max())

        if lo == hi:
            # all MI equal => single bin
            return [BinInfo(lo, hi, mi_list[:])]

        edges = np.linspace(lo, hi, num_bins + 1)

        out: List[BinInfo] = []
        for i in range(num_bins):
            a, b = edges[i], edges[i + 1]
            items = [
                (p, mi) for (p, mi) in mi_list
                if (a <= mi < b) or (i == num_bins - 1 and mi == b)
            ]
            if items:
                mi_vals = [mi for _, mi in items]
                out.append(BinInfo(min(mi_vals), max(mi_vals), items))

        return out

    # --------------------------------------------------
    # Adaptive histogram: try 4 -> 3 -> 2
    # Accept only if every bin has >= min_bin
    # --------------------------------------------------
    def _adaptive_histogram(self, mi_list: List[Tuple[str, float]]) -> Optional[List[BinInfo]]:
        for b in range(self.start_bins, 1, -1):
            bins = self._histogram_bins(mi_list, b)

            # IMPORTANT: bins count can drop due to empty bins => must still respect min_bin
            if len(bins) <= 1:
                continue

            if all(len(x.items) >= self.min_bin for x in bins):
                return bins

        return None

    # --------------------------------------------------
    # Forced Quantile-3 (MI-based) for histogram failure
    # --------------------------------------------------
    def _force_quantile_3(self, mi_list: List[Tuple[str, float]]) -> List[BinInfo]:
        mi_sorted = sorted(mi_list, key=lambda x: x[1])
        n = len(mi_sorted)

        if n == 0:
            return []

        if n < 3:
            vals = [mi for _, mi in mi_sorted]
            return [BinInfo(min(vals), max(vals), mi_sorted)]

        c1 = n // 3
        c2 = 2 * n // 3

        def mk(chunk):
            vals = [mi for _, mi in chunk]
            return BinInfo(min(vals), max(vals), chunk)

        return [mk(mi_sorted[:c1]), mk(mi_sorted[c1:c2]), mk(mi_sorted[c2:])]

    # --------------------------------------------------
    # Low-sample strategy (ONLY when class is small from start)
    # random or MI quantile-2
    # --------------------------------------------------
    def _low_sample_split(self, images: List[str], out: str, cls: str):
        """
        Called ONLY when len(images) < min_bin (class is small from the beginning).
        Strategy is user-controlled:
          - random
          - mi_quantile (2 parts)
        Always keeps reference in train.
        """
        if len(images) == 0:
            return

        ref = images[0]

        # ---- RANDOM ----
        if self.low_sample_strategy == "random":
            paths = images[:]  # include ref
            random.shuffle(paths)
            self._copy_split(paths, out, cls)
            # ensure ref in train
            shutil.copy(ref, os.path.join(out, "train", cls))
            return

        # ---- MI QUANTILE-2 ----
        ref_img = load_image(ref, None if self.strict_shape else self.resize_to)

        mi_list = []
        for p in images[1:]:
            try:
                img = load_image(p, None if self.strict_shape else self.resize_to)
                mi = compute_mi(ref_img, img, self.mode)
                if not np.isnan(mi):
                    mi_list.append((p, float(mi)))
            except Exception:
                continue

        # if MI fails, fallback to random (still low-sample case allowed)
        if len(mi_list) < 2:
            paths = images[:]
            random.shuffle(paths)
            self._copy_split(paths, out, cls)
            shutil.copy(ref, os.path.join(out, "train", cls))
            return

        mi_sorted = sorted(mi_list, key=lambda x: x[1])
        mid = len(mi_sorted) // 2

        low = [p for p, _ in mi_sorted[:mid]]
        high = [p for p, _ in mi_sorted[mid:]]

        random.shuffle(low)
        random.shuffle(high)

        paths = [ref] + low + high
        self._copy_split(paths, out, cls)
        shutil.copy(ref, os.path.join(out, "train", cls))

    # --------------------------------------------------
    # Main
    # --------------------------------------------------
    def run(self, output_root: str):
        for cls in os.listdir(self.dataset_root):
            class_dir = os.path.join(self.dataset_root, cls)
            if not os.path.isdir(class_dir):
                continue

            self._ensure_dirs(output_root, cls)

            images = self._list_images(class_dir)
            if self.verbose:
                print(f"\n▶ Processing class: {cls}")
                print(f"  Found {len(images)} images")

            if len(images) == 0:
                if self.verbose:
                    print("  ⚠ Skipped: no images found.")
                continue

            # -----------------------------
            # CASE A: class is small from the start
            # -----------------------------
            if len(images) <= self.min_bin:
                if self.verbose:
                    print(f"  ⚠ Class too small (<= min_bin={self.min_bin}) → low_sample_strategy='{self.low_sample_strategy}'")
                self._low_sample_split(images, output_root, cls)
                continue

            # -----------------------------
            # CASE B: class is big -> do MI + histogram
            # -----------------------------
            ref = images[0]
            ref_img = load_image(ref, None if self.strict_shape else self.resize_to)

            mi_list: List[Tuple[str, float]] = []
            for p in tqdm(images[1:], desc="  Computing MI", disable=not self.verbose):
                try:
                    img = load_image(p, None if self.strict_shape else self.resize_to)
                    mi = compute_mi(ref_img, img, self.mode)
                    if not np.isnan(mi):
                        mi_list.append((p, float(mi)))
                except Exception:
                    continue

            if self.verbose:
                print(f"  Valid MI values: {len(mi_list)} / {len(images) - 1}")

            # If MI list is too small, we can't histogram reliably -> FORCED Quantile-3
            # (because class was big; here random is NOT allowed)
            if len(mi_list) < 3:
                if self.verbose:
                    print("  ⚠ Too few MI values for histogram → FORCED Quantile-3")
                bins = self._force_quantile_3(mi_list)
            else:
                bins = self._adaptive_histogram(mi_list)

                # -----------------------------
                # CASE C: histogram failed 4->3->2
                # must be FORCED Quantile-3 (no random)
                # -----------------------------
                if bins is None:
                    if self.verbose:
                        print("  ⚠ Histogram failed (4→3→2) → FORCED Quantile-3 (MI-based)")
                    bins = self._force_quantile_3(mi_list)

            if self.verbose and bins is not None:
                sizes = [len(b.items) for b in bins]
                print(f"  ✓ Using bins={len(bins)} | sizes={sizes}")

            # split each bin (ONLY MI images)
            # NOTE: ref is not in bins; we'll copy it once at end.
            if bins:
                for b in bins:
                    paths = [p for p, _ in b.items]
                    random.shuffle(paths)
                    self._copy_split(paths, output_root, cls)

            # add reference image once
            shutil.copy(ref, os.path.join(output_root, "train", cls))

        if self.verbose:
            print("\n✅ MIGT histogram-based splitting finished.")
