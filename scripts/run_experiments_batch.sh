#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

${PYTHON_BIN} scripts/run_experiments_batch.py "$@"
