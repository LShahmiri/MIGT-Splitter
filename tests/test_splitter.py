import csv
import os
import warnings

import cv2
import numpy as np
import pytest

from migt import MIGTSplitter, class_counts, split_counts, stratified_counts

SIZES = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 20, 33, 47, 80]


def write_class(root, name, n, seed, size=(24, 24)):
    rng = np.random.default_rng(seed)
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([127 + 120 * np.sin(xx / (2 + seed % 3)),
                     127 + 120 * np.cos(yy / (3 + seed % 2)),
                     127 + 120 * np.sin((xx + yy) / 4.0)], axis=-1)
    for i in range(n):
        a = 0.15 + 0.8 * rng.random()
        img = np.clip(a * base + (1 - a) * rng.integers(0, 256, (h, w, 3)), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(d, f"img_{i:03d}.png"), img)


@pytest.fixture
def make_ds(tmp_path):
    counter = {"i": 0}

    def _make(spec):
        counter["i"] += 1
        root = str(tmp_path / f"ds{counter['i']}")
        for k, (name, n) in enumerate(sorted(spec.items())):
            write_class(root, name, n, seed=k + 1)
        return root

    return _make


def tree(root):
    out = {}
    for split in ("train", "test", "val"):
        d = os.path.join(root, split)
        if os.path.isdir(d):
            for cls in os.listdir(d):
                for f in os.listdir(os.path.join(d, cls)):
                    out.setdefault((cls, f), []).append(split)
    return out


def quiet(**kw):
    return dict(verbose=False, **kw)


# ------------------------------------------------------------------ counts
def test_class_counts_examples():
    assert class_counts(4, [0.7, 0.2, 0.1]) == [2, 1, 1]
    assert class_counts(9, [0.7, 0.2, 0.1]) == [6, 2, 1]
    assert class_counts(2, [0.7, 0.2, 0.1]) == [1, 1, 0]
    assert split_counts(9, [0.7, 0.2, 0.1]) == [8, 1, 0]          # old per-bin rounding
    assert stratified_counts([4, 5, 5], [0.7, 0.2, 0.1], 1) == [[3, 1, 0], [3, 1, 1], [4, 1, 0]]


# ------------------------------------------------------------------ integrity
@pytest.mark.parametrize("rounding", ["class", "per_bin"])
@pytest.mark.parametrize("strategy", ["random", "mi_quantile"])
@pytest.mark.parametrize("val", [None, 0.1])
def test_no_loss_no_leakage(make_ds, tmp_path, rounding, strategy, val):
    root = make_ds({f"c{n:02d}": n for n in SIZES})
    train, test = (0.8, 0.2) if val is None else (0.7, 0.2)
    out = str(tmp_path / "out")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MIGTSplitter(root, train=train, test=test, val=val, low_sample_strategy=strategy,
                           rounding=rounding, **quiet()).run(out)
    copied = tree(out)
    assert all(len(s) == 1 for s in copied.values())                     # no image in two splits
    assert len(copied) == sum(SIZES)                                     # no image lost
    splits = ["train", "test"] + (["val"] if val else [])
    for n in SIZES:
        rec = res[f"c{n:02d}"]
        if rec["reference"] is not None:                                 # reference only in train
            ref = os.path.basename(rec["reference"])
            assert copied[(f"c{n:02d}", ref)] == ["train"]
        if rounding == "class" and n >= len(splits):
            got = {s: sum(1 for (c, _), v in copied.items() if c == f"c{n:02d}" and v == [s]) for s in splits}
            assert min(got.values()) >= 1, (n, rec["strategy"], got)    # no empty split


def test_class_sizes_follow_ratios(make_ds):
    root = make_ds({f"c{n:02d}": n for n in SIZES})
    res = MIGTSplitter(root, train=0.7, test=0.2, val=0.1, **quiet()).plan()
    for n in SIZES:
        items = res[f"c{n:02d}"]["items"]
        got = [sum(it["split"] == s for it in items) for s in ("train", "test", "val")]
        assert got == class_counts(n, [0.7, 0.2, 0.1]), (n, got)


def test_defaults_follow_the_paper(make_ds):
    root = make_ds({"small": 9, "eq": 10, "big": 60})
    s = MIGTSplitter(root, **quiet())
    assert (s.min_bin, s.low_sample_strategy, s.rounding, s.unscored) == (10, "random", "class", "include")
    res = s.plan()
    assert res["small"]["strategy"] == res["eq"]["strategy"] == "low_sample_random"   # <= min_bin
    assert res["big"]["strategy"] in ("histogram", "forced_quantile")


def test_random_small_class_has_no_fixed_reference(make_ds):
    root = make_ds({"small": 9})
    first_split = set()
    for seed in range(20):
        rec = MIGTSplitter(root, train=0.7, test=0.2, val=0.1, seed=seed, **quiet()).plan()["small"]
        assert rec["reference"] is None and len(rec["items"]) == 9
        first_split.add(next(it["split"] for it in rec["items"] if it["path"].endswith("img_000.png")))
    assert first_split != {"train"}


def test_mi_quantile_small_class_is_stratified(make_ds):
    root = make_ds({"small": 10})
    low_in_test = False
    for seed in range(20):
        rec = MIGTSplitter(root, train=0.5, test=0.4, val=0.1, low_sample_strategy="mi_quantile",
                           seed=seed, **quiet()).plan()["small"]
        assert rec["strategy"] == "low_sample_mi_quantile"
        low_in_test |= any(it["bin"] == 0 and it["split"] != "train" for it in rec["items"])
    assert low_in_test        # test/val also come from the low-MI half


# ------------------------------------------------------------------ unscored images
def _add_bad_images(root):
    d = os.path.join(root, "c")
    cv2.imwrite(os.path.join(d, "zz_big.png"), np.zeros((40, 30, 3), np.uint8) + 7)
    with open(os.path.join(d, "zz_broken.jpg"), "wb") as f:
        f.write(b"broken")


def test_unscored_images_are_kept_and_reported(make_ds, tmp_path):
    root = make_ds({"c": 30})
    _add_bad_images(root)
    out = str(tmp_path / "out")
    with pytest.warns(UserWarning, match=r"2 image\(s\) could not be MI-scored and were split at random"):
        res = MIGTSplitter(root, **quiet()).run(out)
    assert len(tree(out)) == 32
    roles = {os.path.basename(it["path"]): it for it in res["c"]["items"]}
    assert roles["zz_big.png"]["role"] == roles["zz_broken.jpg"]["role"] == "unscored"
    assert "shape" in roles["zz_big.png"]["reason"].lower()
    rows = list(csv.DictReader(open(os.path.join(out, "migt_manifest.csv"), encoding="utf-8")))
    assert len(rows) == 32 and sum(r["role"] == "unscored" for r in rows) == 2


def test_skip_mode_warns(make_ds, tmp_path):
    root = make_ds({"c": 30})
    _add_bad_images(root)
    out = str(tmp_path / "out")
    with pytest.warns(UserWarning, match=r"2 image\(s\) could not be MI-scored and are not in any split"):
        res = MIGTSplitter(root, unscored="skip", **quiet()).run(out)
    assert len(tree(out)) == 30 and len(res["c"]["skipped"]) == 2


def test_grayscale_photo_in_colour_class_is_scored(make_ds):
    root = make_ds({"c": 30})
    p = os.path.join(root, "c", "img_005.png")
    cv2.imwrite(p, cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY))
    rec = MIGTSplitter(root, **quiet()).plan()["c"]
    it = next(i for i in rec["items"] if i["path"].endswith("img_005.png"))
    assert it["role"] == "sample" and 1.0 <= it["mi"] <= 2.0


def test_non_ascii_path(tmp_path):
    root = str(tmp_path / "données")
    write_class(root, "papillon_é", 25, seed=1)
    rec = MIGTSplitter(root, **quiet()).plan()["papillon_é"]
    assert all(it["role"] != "unscored" for it in rec["items"])


# ------------------------------------------------------------------ output and reproducibility
def test_output_reuse_and_inside_dataset_refused(make_ds, tmp_path):
    root = make_ds({"c": 30})
    out = str(tmp_path / "out")
    MIGTSplitter(root, seed=1, **quiet()).run(out)
    with pytest.raises(ValueError, match="already contains images"):
        MIGTSplitter(root, seed=2, **quiet()).run(out)
    with pytest.raises(ValueError, match="must not be dataset_root"):
        MIGTSplitter(root, **quiet()).run(os.path.join(root, "out"))


def test_reproducible_and_seed_dependent(make_ds):
    root = make_ds({"a": 60, "b": 9})

    def assign(seed):
        res = MIGTSplitter(root, seed=seed, **quiet()).plan()
        return {(c, it["path"]): it["split"] for c, r in res.items() for it in r["items"]}

    assert assign(3) == assign(3) and assign(3) != assign(4)


def test_resize_ignored_warning(make_ds):
    root = make_ds({"c": 12})
    with pytest.warns(UserWarning, match="ignored"):
        MIGTSplitter(root, resize_to=(10, 10), **quiet())


@pytest.mark.parametrize("kw", [{"low_sample_strategy": "x"}, {"rounding": "x"}, {"unscored": "x"},
                                {"train": 0.8, "test": 0.3}, {"strict_shape": False}])
def test_invalid_parameters(make_ds, kw):
    root = make_ds({"c": 12})
    with pytest.raises(ValueError):
        MIGTSplitter(root, **quiet(**kw))
