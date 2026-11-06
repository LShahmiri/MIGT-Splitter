# Changelog

## [1.2.0] - 2026-11-06

### Changed
- `min_bin` defaults to 10, the value used in the paper (was 13).
- `low_sample_strategy` defaults to `"random"`: classes with `<= min_bin`
  images are split at random, as in the paper. In a random split the
  reference image has no special role.
- With `low_sample_strategy="mi_quantile"` the two MI halves are used as
  strata, so test/validation images come from both halves (before, the
  sequence reference + low half + high half was cut in this order and
  test/val came only from the high-MI half).
- Train/test/val sizes are rounded once per class (largest remainder, at
  least one image per split when the class has at least as many images as
  splits) and spread over the MI bins in proportion to their sizes
  (`rounding="class"`). `rounding="per_bin"` rounds every bin separately,
  as before.
- Classes and files are processed in sorted order with a private, seeded
  random generator, so results no longer depend on the file system or on
  the global random state.
- `mode="auto"` uses the reference image to choose grayscale or colour MI;
  a grayscale photo in a colour class is replicated to three channels
  instead of failing.

### Added
- `unscored` option: images whose MI cannot be computed (unreadable, other
  size with `strict_shape=True`) are split at random as an extra stratum of
  their class (`"include"`, default) or left out (`"skip"`); both raise a
  `UserWarning`.
- `plan()` computes the partition without copying; `run()` returns it.
- `migt_manifest.csv` in the output folder lists every image with its
  split, role, MI score and bin.
- `class_counts()` and `stratified_counts()`.
- Tests (`pytest`), LICENSE and this changelog.

### Fixed
- Images whose MI failed were left out of every split without any message.
- With `low_sample_strategy="random"` the reference image could be copied
  into both train and test.
- A class could end with no test or validation image because every bin was
  rounded separately.
- `run()` refuses an output folder that already contains images, so a
  second run cannot put an image into two splits.
- Paths with non-ASCII characters can be read; an alpha channel is dropped.
- A warning is shown when `resize_to` is ignored because `strict_shape=True`.
