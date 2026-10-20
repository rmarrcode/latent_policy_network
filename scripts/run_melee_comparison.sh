#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
exec "$PYTHON_BIN" -m latent_policy.melee_self_play "$@"
