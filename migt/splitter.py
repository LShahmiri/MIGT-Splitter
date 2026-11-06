import csv
import os
import random
import shutil
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .mi import compute_mi, is_grayscale, load_image
from .utils import (
    IMAGE_EXTENSIONS,
    ROUNDING_MODES,
    list_classes,
    list_images,
    split_counts,
    stratified_counts,
)

LOW_SAMPLE_STRATEGIES = ("random", "mi_quantile")
UNSCORED_MODES = ("include", "skip")
MANIFEST_FILENAME = "migt_manifest.csv"


# --------------------------------------------------
# Bin structure
# --------------------------------------------------
@dataclass
class BinInfo:
    mi_min: float
    mi_max: float
    items: List[Tuple[str, float]]  # (path, mi)


def histogram_bins(mi_list, num_bins):
    """Equal-width bins over [min(MI), max(MI)], intervals [a, b) and a closed last bin.
    Only non-empty bins are returned; equal MI values give a single bin."""
    if len(mi_list) == 0:
        return []
    values = np.array([mi for _, mi in mi_list], dtype=float)
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return [BinInfo(lo, hi, list(mi_list))]
    edges = np.linspace(lo, hi, num_bins + 1)
    idx = np.searchsorted(edges[1:-1], values, side="right")
    out = []
    for i in range(num_bins):
        items = [item for item, k in zip(mi_list, idx) if k == i]
        if items:
            vals = [mi for _, mi in items]
            out.append(BinInfo(min(vals), max(vals), items))
    return out


def adaptive_histogram(mi_list, start_bins, min_bin):
    """Try start_bins -> ... -> 2 bins; accept the first histogram whose bins all hold >= min_bin."""
    for b in range(start_bins, 1, -1):
        bins = histogram_bins(mi_list, b)
        if len(bins) <= 1:  # bins can drop out when empty
            continue
        if all(len(x.items) >= min_bin for x in bins):
            return bins
    return None


def forced_quantile_split(mi_list, n_parts=3):
    """Sort by MI and cut into n_parts equal portions (one portion if fewer items)."""
    mi_sorted = sorted(mi_list, key=lambda x: x[1])
    n = len(mi_sorted)
    if n == 0:
        return []
    if n < n_parts:
        vals = [mi for _, mi in mi_sorted]
        return [BinInfo(min(vals), max(vals), mi_sorted)]
    cuts = [i * n // n_parts for i in range(n_parts + 1)]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        chunk = mi_sorted[a:b]
        vals = [mi for _, mi in chunk]
        out.append(BinInfo(min(vals), max(vals), chunk))
    return out


# --------------------------------------------------
# MIGT Splitter
# --------------------------------------------------
class MIGTSplitter:
    """
    Mutual Information Guided Training (MIGT) dataset partitioning.

    For every class the first image (sorted by file name) is the reference.
    The MI between the reference and every other image is computed, the MI
    values are binned with an adaptive histogram (bins -> ... -> 2, every bin
    must hold at least min_bin images; otherwise three equal MI portions are
    used), and every bin is split by the train/test/val ratios. The reference
    image is always in train. Classes with <= min_bin images are split at
    random (low_sample_strategy="random", the default) or by two MI halves
    ("mi_quantile").

    rounding="class" (default): the split sizes are rounded once per class and
    spread over the bins, so no split is left empty by rounding.
    rounding="per_bin": every bin is rounded separately (versions <= 1.1.1).

    unscored="include" (default): images whose MI cannot be computed are split
    at random as an extra stratum of their class, with a warning.
    unscored="skip": they are left out, with a warning.
    """

    def __init__(
        self,
        dataset_root: str,
        mode: str = "auto",
        bins: int = 4,
        min_bin: int = 10,
        train: float = 0.8,
        test: float = 0.2,
        val: Optional[float] = None,
        strict_shape: bool = True,
        resize_to: Optional[Tuple[int, int]] = None,
        seed: int = 42,
        verbose: bool = True,
        low_sample_strategy: str = "random",  # "random" | "mi_quantile"
        rounding: str = "class",               # "class" | "per_bin"
        unscored: str = "include",             # "include" | "skip"
    ):
        self.dataset_root = os.fspath(dataset_root)
        self.mode = mode
        self.start_bins = int(bins)
        self.min_bin = int(min_bin)
        self.strict_shape = strict_shape
        self.resize_to = tuple(resize_to) if resize_to is not None else None
        self.seed = seed
        self.verbose = verbose
        self.low_sample_strategy = low_sample_strategy
        self.rounding = rounding
        self.unscored = unscored

        if not os.path.isdir(self.dataset_root):
            raise ValueError(f"dataset_root is not a directory: {self.dataset_root!r}")
        if mode not in ("auto", "grayscale", "color"):
            raise ValueError("mode must be 'auto', 'grayscale' or 'color'.")
        if self.start_bins < 2:
            raise ValueError("bins must be >= 2.")
        if self.min_bin < 1:
            raise ValueError("min_bin must be >= 1.")
        if not strict_shape and resize_to is None:
            raise ValueError("When strict_shape=False, resize_to must be provided, e.g. resize_to=(224,224).")
        if strict_shape and resize_to is not None:
            warnings.warn("resize_to is ignored because strict_shape=True.", UserWarning, stacklevel=2)
        if low_sample_strategy not in LOW_SAMPLE_STRATEGIES:
            raise ValueError("low_sample_strategy must be 'random' or 'mi_quantile'.")
        if rounding not in ROUNDING_MODES:
            raise ValueError(f"rounding must be one of {ROUNDING_MODES}.")
        if unscored not in UNSCORED_MODES:
            raise ValueError(f"unscored must be one of {UNSCORED_MODES}.")

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

    # --------------------------------------------------
    # Utils
    # --------------------------------------------------
    @property
    def split_names(self):
        return ["train", "test", "val"][:len(self.ratios)]

    def _log(self, msg):
        if self.verbose:
            print(msg)

    @property
    def _target_size(self):
        return None if self.strict_shape else self.resize_to

    def _assign(self, paths):
        """Split paths in order by the ratios (rounding="per_bin")."""
        counts = split_counts(len(paths), self.ratios)
        n_train, n_test = counts[0], counts[1]
        out = {}
        for i, p in enumerate(paths):
            if i < n_train:
                out[p] = "train"
            elif i < n_train + n_test:
                out[p] = "test"
            else:
                out[p] = "val" if self.val is not None else "test"
        return out

    def _split_strata(self, strata, rng, ref=None):
        """Shuffle every stratum and assign its images; ref (if given) goes to train."""
        shuffled = []
        for paths in strata:
            p = list(paths)
            rng.shuffle(p)
            shuffled.append(p)
        out = {ref: "train"} if ref is not None else {}
        if self.rounding == "per_bin":
            for p in shuffled:
                out.update(self._assign(p))
            return out
        counts = stratified_counts([len(p) for p in shuffled], self.ratios, 1 if ref is not None else 0)
        for p, c in zip(shuffled, counts):
            k = 0
            for split, m in zip(self.split_names, c):
                out.update({x: split for x in p[k:k + m]})
                k += m
        return out

    def _reference_image(self, ref, cls):
        try:
            ref_img = load_image(ref, self._target_size)
        except ValueError as exc:
            raise ValueError(f"Reference image of class {cls!r} cannot be loaded: {exc}") from exc
        if self.mode == "color" and is_grayscale(ref_img):
            raise ValueError(f"mode='color' requires colour images, but the reference image of class "
                             f"{cls!r} is grayscale: {ref}. Use mode='auto' or mode='grayscale'.")
        return ref_img

    def _scores(self, ref_img, paths):
        """[(path, mi or None, reason or None)] for every path."""
        out = []
        for p in tqdm(paths, desc="  Computing MI", disable=not self.verbose):
            try:
                img = load_image(p, self._target_size)
                if img.shape[:2] != ref_img.shape[:2]:
                    raise ValueError(f"Image shape mismatch: {ref_img.shape} vs {img.shape}")
                mi = compute_mi(ref_img, img, self.mode)
                if np.isnan(mi):
                    out.append((p, None, "MI is NaN"))
                else:
                    out.append((p, float(mi), None))
            except Exception as exc:  # noqa: BLE001 - reported, never dropped silently
                out.append((p, None, f"{type(exc).__name__}: {exc}"))
        return out

    def _unscored_paths(self, scores, rec):
        failed = [(p, reason) for p, mi, reason in scores if mi is None]
        if self.unscored == "skip":
            rec["skipped"] = [{"path": p, "reason": r} for p, r in failed]
            return []
        return [p for p, _ in failed]

    @staticmethod
    def _items(assigned, paths, role="sample", mi=None, bin_of=None, reasons=None):
        return [{"path": p, "split": assigned[p], "role": role,
                 "mi": (mi or {}).get(p), "bin": (bin_of or {}).get(p),
                 "reason": (reasons or {}).get(p)} for p in paths]

    # --------------------------------------------------
    # Low-sample strategy (ONLY when class is small from start)
    # --------------------------------------------------
    def _low_sample_split(self, cls, images, rec, rng):
        ref, others = images[0], images[1:]

        def random_split(strategy):
            rec["strategy"] = strategy
            if self.rounding == "per_bin":
                paths = others[:]
                rng.shuffle(paths)
                assigned = self._assign([ref] + paths)   # reference in the first train slot
                rec["reference"] = ref
                rec["items"] = self._items(assigned, [ref], "reference") + self._items(assigned, paths)
                return
            assigned = self._split_strata([images], rng)
            rec["items"] = self._items(assigned, images)

        if self.low_sample_strategy == "random":
            random_split("low_sample_random")
            return

        ref_img = self._reference_image(ref, cls)
        scores = self._scores(ref_img, others)
        mi = {p: v for p, v, _ in scores if v is not None}
        reasons = {p: r for p, v, r in scores if v is None}
        mi_list = [(p, v) for p, v, _ in scores if v is not None]
        if len(mi_list) < 2:
            self._log("  Too few MI values -> random split")
            random_split("low_sample_random_fallback")
            return

        unscored = self._unscored_paths(scores, rec)
        halves = forced_quantile_split(mi_list, 2)
        rec["strategy"] = "low_sample_mi_quantile"
        rec["reference"] = ref
        rec["bins"] = [len(h.items) for h in halves]
        if self.rounding == "per_bin":
            ordered, bin_of = [], {}
            for i, h in enumerate(halves):
                paths = [p for p, _ in h.items]
                rng.shuffle(paths)
                ordered += paths
                bin_of.update({p: i for p in paths})
            assigned = self._assign([ref] + ordered)
            if unscored:
                assigned.update(self._split_strata([unscored], rng))
        else:
            strata = [[p for p, _ in h.items] for h in halves]
            bin_of = {p: i for i, s in enumerate(strata) for p in s}
            assigned = self._split_strata(strata + ([unscored] if unscored else []), rng, ref)
        rec["items"] = (self._items(assigned, [ref], "reference")
                        + self._items(assigned, [p for p in others if p in bin_of], mi=mi, bin_of=bin_of)
                        + self._items(assigned, unscored, "unscored", reasons=reasons))

    # --------------------------------------------------
    # One class
    # --------------------------------------------------
    def _plan_class(self, cls, rng):
        images = list_images(os.path.join(self.dataset_root, cls))
        rec = {"class": cls, "n_images": len(images), "strategy": "empty", "reference": None,
               "bins": [], "items": [], "skipped": []}
        self._log(f"\n> Processing class: {cls}")
        self._log(f"  Found {len(images)} images")
        if not images:
            self._log("  Skipped: no images found.")
            return rec

        # CASE A: class is small from the start
        if len(images) <= self.min_bin:
            self._log(f"  Class too small (<= min_bin={self.min_bin}) -> "
                      f"low_sample_strategy='{self.low_sample_strategy}'")
            self._low_sample_split(cls, images, rec, rng)
            return rec

        # CASE B: class is big -> MI + histogram
        ref = images[0]
        ref_img = self._reference_image(ref, cls)
        scores = self._scores(ref_img, images[1:])
        mi = {p: v for p, v, _ in scores if v is not None}
        reasons = {p: r for p, v, r in scores if v is None}
        mi_list = [(p, v) for p, v, _ in scores if v is not None]
        unscored = self._unscored_paths(scores, rec)
        self._log(f"  Valid MI values: {len(mi_list)} / {len(images) - 1}")
        if reasons:
            what = "split at random as an extra stratum" if unscored else "skipped"
            self._log(f"  {len(reasons)} image(s) whose MI could not be computed are {what} "
                      f"(e.g. {next(iter(reasons.values()))})")

        if len(mi_list) < 3:
            self._log("  Too few MI values for histogram -> FORCED Quantile-3")
            bins, strategy = forced_quantile_split(mi_list, 3), "forced_quantile"
        else:
            bins = adaptive_histogram(mi_list, self.start_bins, self.min_bin)
            strategy = "histogram"
            # CASE C: histogram failed bins -> 2: FORCED quantile split (no random)
            if bins is None:
                self._log(f"  Histogram failed ({self.start_bins}->2) -> FORCED Quantile-3 (MI-based)")
                bins, strategy = forced_quantile_split(mi_list, 3), "forced_quantile"
        self._log(f"  Using bins={len(bins)} | sizes={[len(b.items) for b in bins]}")

        strata = [[p for p, _ in b.items] for b in bins]
        bin_of = {p: i for i, s in enumerate(strata) for p in s}
        assigned = self._split_strata(strata + ([unscored] if unscored else []), rng, ref)
        rec.update(strategy=strategy, reference=ref, bins=[len(s) for s in strata])
        rec["items"] = (self._items(assigned, [ref], "reference")
                        + self._items(assigned, [p for s in strata for p in s], mi=mi, bin_of=bin_of)
                        + self._items(assigned, unscored, "unscored", reasons=reasons))
        return rec

    # --------------------------------------------------
    # Main
    # --------------------------------------------------
    def plan(self) -> Dict[str, dict]:
        """
        Compute the partition without copying files. Returns {class: record} with
        the strategy, bin sizes and, for every image, its split, role, MI and bin.
        Checks that every image is in exactly one split (or listed as skipped).
        """
        rng = random.Random(self.seed)
        result = {cls: self._plan_class(cls, rng) for cls in list_classes(self.dataset_root)}
        for cls, rec in result.items():
            listed = [it["path"] for it in rec["items"]] + [s["path"] for s in rec["skipped"]]
            if len(listed) != rec["n_images"] or len(set(listed)) != len(listed):
                raise RuntimeError(f"Class {cls!r}: images lost or duplicated in the partition")

        skipped = [(c, s) for c, r in result.items() for s in r["skipped"]]
        if skipped:
            warnings.warn(f"{len(skipped)} image(s) could not be MI-scored and are not in any split, e.g. "
                          f"{skipped[0][1]['path']} ({skipped[0][1]['reason']}).", UserWarning, stacklevel=2)
        unscored = [(c, it) for c, r in result.items() for it in r["items"] if it["role"] == "unscored"]
        if unscored:
            warnings.warn(f"{len(unscored)} image(s) could not be MI-scored and were split at random as an "
                          f"extra stratum of their class, e.g. {unscored[0][1]['path']} "
                          f"({unscored[0][1]['reason']}).", UserWarning, stacklevel=2)
        empty = []
        for c, r in result.items():
            counts = Counter(it["split"] for it in r["items"])
            if r["n_images"] >= len(self.split_names):
                empty += [f"{c} ({s})" for s in self.split_names if counts[s] == 0]
        if empty:
            warnings.warn(f"No image assigned to: {', '.join(empty[:5])}{' ...' if len(empty) > 5 else ''}.",
                          UserWarning, stacklevel=2)
        return result

    def run(self, output_root: str) -> Dict[str, dict]:
        """
        Partition the dataset and copy the original images to
        output_root/train/<class>/, output_root/test/<class>/ (and val/).
        A manifest of every image (migt_manifest.csv) is written to output_root.
        The output folders must not already contain images.
        """
        out = os.path.abspath(os.fspath(output_root))
        root = os.path.abspath(self.dataset_root)
        if out == root or out.startswith(root + os.sep):
            raise ValueError("output_root must not be dataset_root or lie inside it.")
        result = self.plan()

        for cls in result:
            for s in self.split_names:
                d = os.path.join(out, s, cls)
                if os.path.isdir(d) and any(f.lower().endswith(IMAGE_EXTENSIONS) for f in os.listdir(d)):
                    raise ValueError(f"{d!r} already contains images (an earlier run?). Use an empty "
                                     "output_root so that no image ends up in two splits.")

        for cls, rec in result.items():
            for s in self.split_names:
                os.makedirs(os.path.join(out, s, cls), exist_ok=True)
            for it in rec["items"]:
                shutil.copy(it["path"], os.path.join(out, it["split"], cls))
            for s in self.split_names:
                planned = sorted(os.path.basename(it["path"]) for it in rec["items"] if it["split"] == s)
                copied = sorted(f for f in os.listdir(os.path.join(out, s, cls))
                                if f.lower().endswith(IMAGE_EXTENSIONS))
                if planned != copied:
                    raise RuntimeError(f"{os.path.join(out, s, cls)!r}: copied files do not match the plan "
                                       "(duplicate file names or a failed copy)")

        with open(os.path.join(out, MANIFEST_FILENAME), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "file", "split", "role", "strategy", "mi", "bin", "reason"])
            for cls, rec in result.items():
                for it in rec["items"]:
                    w.writerow([cls, os.path.basename(it["path"]), it["split"], it["role"], rec["strategy"],
                                "" if it["mi"] is None else it["mi"], "" if it["bin"] is None else it["bin"],
                                it["reason"] or ""])
                for s in rec["skipped"]:
                    w.writerow([cls, os.path.basename(s["path"]), "", "skipped", rec["strategy"], "", "",
                                s["reason"]])

        if self.verbose:
            n = sum(r["n_images"] for r in result.values())
            n_assigned = sum(len(r["items"]) for r in result.values())
            print(f"\nMIGT histogram-based splitting finished: {n} image(s) found, {n_assigned} assigned, "
                  f"{n - n_assigned} skipped. Manifest: {os.path.join(out, MANIFEST_FILENAME)}")
        return result
