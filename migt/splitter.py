import os
import shutil
import random
from tqdm import tqdm

from .mi import load_image, compute_mi
from .utils import split_counts


class MIGTSplitter:
    def __init__(
        self,
        dataset_root,
        mode="auto",
        bins=4,
        min_bin=10,
        train=0.5,
        test=0.4,
        val=0.1,
        seed=42,
    ):
        self.dataset_root = dataset_root
        self.mode = mode
        self.bins = bins
        self.min_bin = min_bin
        self.ratios = [train, test, val]
        self.seed = seed



    def _ensure_dirs(self, output_root, cls):
        for split in ["train", "test", "val"]:
            os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)

    def run(self, output_root):
        classes = [
            d for d in os.listdir(self.dataset_root)
            if os.path.isdir(os.path.join(self.dataset_root, d))
        ]

        for cls in classes:
            print(f"\n▶ Processing class: {cls}")
            self._ensure_dirs(output_root, cls)

            class_dir = os.path.join(self.dataset_root, cls)
            images = sorted([
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])

            ref_img = load_image(images[0])

            mi_list = []
            for p in tqdm(images):
                img = load_image(p)
                mi = compute_mi(ref_img, img, self.mode)
                mi_list.append((p, mi))

            mi_list.sort(key=lambda x: x[1])

            # ---------- Histogram binning ----------
            num_bins = self.bins
            bins_data = None

            while num_bins >= 2:
                size = len(mi_list) // num_bins
                tmp = [
                    mi_list[i * size:(i + 1) * size]
                    for i in range(num_bins)
                ]
                if all(len(b) >= self.min_bin for b in tmp):
                    bins_data = tmp
                    break
                num_bins -= 1

            # ---------- Fallback ----------
            if bins_data is None:
                part = len(mi_list) // 3
                bins_data = [
                    mi_list[:part],
                    mi_list[part:2 * part],
                    mi_list[2 * part:]
                ]

            # ---------- Split ----------
            for bin_data in bins_data:
                paths = [x[0] for x in bin_data]
                random.shuffle(paths)

                n = len(paths)
                n_train, n_test, n_val = split_counts(n, self.ratios)

                train = paths[:n_train]
                test = paths[n_train:n_train + n_test]
                val = paths[n_train + n_test:]

                for p in train:
                    shutil.copy(p, os.path.join(output_root, "train", cls))
                for p in test:
                    shutil.copy(p, os.path.join(output_root, "test", cls))
                for p in val:
                    shutil.copy(p, os.path.join(output_root, "val", cls))

        print("\n✅ MIGT splitting finished with exact counts.")
