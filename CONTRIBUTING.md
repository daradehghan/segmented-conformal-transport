# Contributing

## Project posture

This is a research software project with an open contribution process.
Well-scoped bug reports, documentation corrections, tests, and targeted
implementation fixes are welcome.

Large feature additions, broad refactors, and workflow changes should be
discussed in advance. That keeps the review scope explicit and reduces
avoidable rework.

## Review process

Issues and pull requests are the public review record for the project. Proposed
changes are reviewed for correctness, reproducibility, documentation
consistency, and scope fit.

Changes that affect paper-facing behavior, benchmark settings, or serialized
state semantics should make that effect explicit in the issue or pull request
discussion.

## Before opening a pull request

Please open an issue first when the proposed change is non-trivial. That keeps
the review scope explicit and reduces avoidable work.

A proposed change should:

- preserve the paper-facing behavior unless the change is an intentional fix
- keep reproducibility paths and documentation synchronized
- avoid introducing local-only artifacts, generated outputs, or raw data into
  version control
- include tests for user-visible behavioral changes

## Local checks

The expected baseline is:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev,plots,detectors]"
PYTHONPATH=src:. pytest -q
```

## Release versioning

The project uses a narrow PEP 440 policy for committed package versions.

Allowed committed forms are:

- `X.Y.Z.devN` for normal development on `main`
- `X.Y.ZrcN` for release candidates
- `X.Y.Z` for final releases

After each final release, `main` should move immediately to the next
development version, for example from `0.2.2` to `0.2.3.dev0`.

The release checker uses two validation modes. Source mode covers local checks,
workflow-dispatch dry runs, branch pushes, and release-prep pull requests. Tag
mode covers pushed release tags and adds tag/version matching plus the rule that
development versions cannot be tagged.

Development versions are source-tree versions only. They are not tagged and are
not published to TestPyPI as part of the normal release workflow. Release
candidates may be tagged as `vX.Y.ZrcN` and publish to TestPyPI plus a GitHub
prerelease. Final releases are tagged as `vX.Y.Z` and publish to PyPI plus a
GitHub release.

Release metadata follows the committed version shape:

- `X.Y.Z.devN` requires a `## Unreleased` changelog heading. `CITATION.cff`
  stays pinned to the latest final citable release and is not required to match
  the development version.
- `X.Y.ZrcN` requires an exact `## X.Y.ZrcN` changelog heading. `CITATION.cff`
  stays pinned to the latest final citable release and is not required to match
  the release-candidate version.
- `X.Y.Z` requires an exact `## X.Y.Z` changelog heading and a matching
  `CITATION.cff` version.

`CHANGELOG.md` is required in all validation modes. `CITATION.cff` is required
when the committed package version is a final `X.Y.Z` release.

Changelog headings used by the release checker must be bare h2 headings with no
date or trailing text: `## Unreleased`, `## X.Y.ZrcN`, or `## X.Y.Z`.

Release-prep pull requests may temporarily set `pyproject.toml` to `X.Y.ZrcN`
or `X.Y.Z` before the tag exists. Those pull requests must include the matching
changelog heading in the same change, and final-release prep must also align
`CITATION.cff` with the final package version.

Do not use `.postN` for ordinary development or maintenance work. Do not commit
local versions such as `+gHASH`; local-version generation is outside the
current release policy.

## Style

Code changes should favor clarity, explicitness, and stable behavior over
cleverness. Documentation, comments, and docstrings should remain formal,
concise, and technically precise.

## Security

Suspected vulnerabilities should not be reported through public issues or pull
requests. Please follow the process in `SECURITY.md`.
