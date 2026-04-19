# Changelog

## 0.2.0 - 2026-04-19

### Added

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
