#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
"$PYTHON_BIN" -m pip install 'melee==0.47.3' \
    'melee-sim-light @ git+https://github.com/kyhavlov/melee-sim-light.git@b8b87dbaaf5f236ae4d9b3aac96ea03d3e49400a'
if [[ $# -gt 0 ]]; then
    "$PYTHON_BIN" -m melee_sim.extract_data --iso "$1" --out-dir "${2:-data}"
fi
