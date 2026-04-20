# Changelog

## 0.2.2rc1 - 2026-04-20

Release candidate for stricter predict/update sequencing enforcement.

### Changed

- `SegmentedTransportCalibrator` now enforces the intended `predict_cdf()` /
  `update()` sequencing contract more strictly.
- `update()` now raises `PredictionSequenceError` for clear sequencing misuse,
  including calls without a pending prediction, repeated updates after a
  prediction has already been consumed, and updates using a forecast id known
  to have been superseded.
- Ambiguous forecast-object mismatches remain warning-only under the existing
  heuristic object-identity check.
- Pending prediction bookkeeping remains transient and is not serialized with
  calibrator state.

## 0.2.1 - 2026-04-20

Correctness and engineering hardening release.

### Fixed

- Made `SegmentedTransportCalibrator.update()` fail atomically when validation
  fails.
- Added explicit rejection of non-finite observed outcomes and invalid PIT
  values.
- Tightened `PageHinkleyDetector` threshold validation to require finite
  positive thresholds.

### CI

- Made mypy type checking blocking in CI and release workflows.

## 0.2.0 - 2026-04-19

### Added

- First public PyPI release of `tsconformal`.
- Tag-driven release automation for TestPyPI, PyPI, and GitHub Releases.
- Release metadata validation via `scripts/check_release_version.py`.
- State-schema compatibility tests backed by a pinned serialized-state fixture.

### Changed

- The package version is now single-source in `pyproject.toml`.
- `tsconformal.__version__` now prefers installed package metadata and falls
  back to `pyproject.toml` during direct source-tree execution.
- Packaging metadata now uses an SPDX license expression with an explicit
  tracked license file.
- Serialized calibrator state now uses an explicit schema-version allowlist
  that is independent of the package version.
- The public docs and installation instructions now reflect the PyPI release
  channel and the new release process.

### Fixed

- Local setup output now derives the installed package version at runtime
  instead of printing a hard-coded string.
- Release publishing now pins `pypa/gh-action-pypi-publish` to the resolved
  release commit so tag-based publication can pull the published container
  image.
