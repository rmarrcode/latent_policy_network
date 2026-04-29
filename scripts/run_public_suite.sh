#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../.venv/bin/python -m latent_policy.public_suite \
  --updates 8 \
  --agents static_mlp hyper_head full_hyper \
  --seeds 1 \
  --no-progress
