# Contributing

## Project posture

This repository is currently maintained as a single-maintainer research software
project. External contributions are therefore limited and remain entirely at the
maintainer's discretion.

Well-scoped bug reports, documentation corrections, and narrowly targeted
improvements are the most useful forms of outside input. Large unsolicited
feature additions, broad refactors, and workflow changes are unlikely to be
accepted unless they are discussed in advance.

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
