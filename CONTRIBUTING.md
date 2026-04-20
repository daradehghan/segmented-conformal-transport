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

## Style

Code changes should favor clarity, explicitness, and stable behavior over
cleverness. Documentation, comments, and docstrings should remain formal,
concise, and technically precise.

## Security

Suspected vulnerabilities should not be reported through public issues or pull
requests. Please follow the process in `SECURITY.md`.
