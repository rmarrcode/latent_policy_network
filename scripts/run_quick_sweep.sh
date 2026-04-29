#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
../.venv/bin/python -m latent_policy.sweep \
  --config configs/quick.yaml \
  --agents static_mlp hyper_head film full_hyper \
  --encoders gru \
  --seeds 1 \
  --updates 12
