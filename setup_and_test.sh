#!/usr/bin/env bash
# setup_and_test.sh — One-command setup for tsconformal
# Usage: bash setup_and_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=== tsconformal v0.1.0 setup ==="

# 1. Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# 2. Upgrade pip
echo "[2/5] Upgrading pip..."
pip install --upgrade pip --quiet

# 3. Install package + dev dependencies
echo "[3/5] Installing tsconformal + dev dependencies..."
pip install -e ".[dev,plots]" --quiet

# 4. Run tests
echo "[4/5] Running tests..."
PYTHONPATH=src pytest tests/ -v --tb=short

# 5. Run example
echo "[5/5] Running synthetic example..."
python src/tsconformal/examples/synthetic_piecewise_stationary.py

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with: source .venv/bin/activate"
echo "Run tests with:                pytest tests/ -v"
echo "Run example with:              python src/tsconformal/examples/synthetic_piecewise_stationary.py"
