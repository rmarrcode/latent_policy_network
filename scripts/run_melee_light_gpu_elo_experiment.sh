#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_TAG="${RUN_TAG:-melee_light_gpu_elo_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-runs/${RUN_TAG}}"
TRAIN_DIR="${TRAIN_DIR:-${BASE_DIR}/train}"
ELO_DIR="${ELO_DIR:-${BASE_DIR}/elo}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"
CONFIG="${CONFIG:-configs/melee_light_gpu_elo.yaml}"
DEVICE="${DEVICE:-cuda}"
TOTAL_UPDATES="${TOTAL_UPDATES:-120}"
SAVE_INTERVAL="${SAVE_INTERVAL:-30}"
SEEDS="${SEEDS:-1 2}"
AGENT_ENCODERS="${AGENT_ENCODERS:-static_mlp:mean hyper_head:gru full_hyper:gru film:gru}"
CHARACTERS="${CHARACTERS:-marth,puff,fox,falco,falcon}"
WARMUP_GAMES="${WARMUP_GAMES:-16}"
SCORED_GAMES="${SCORED_GAMES:-48}"
MAX_PAIRINGS="${MAX_PAIRINGS:-320}"
MIN_PAIRINGS_PER_COMPETITOR="${MIN_PAIRINGS_PER_COMPETITOR:-4}"
ELO_K="${ELO_K:-32}"
ELO_SEED="${ELO_SEED:-20260501}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1}"
LEARNING_RATE="${LEARNING_RATE:-}"
FULL_HYPER_LR="${FULL_HYPER_LR:-0.0001}"
ALLOW_TRAIN_FAILURES="${ALLOW_TRAIN_FAILURES:-1}"

mkdir -p "$TRAIN_DIR" "$ELO_DIR" "$LOG_DIR"

{
  echo "run_tag=${RUN_TAG}"
  echo "base_dir=${BASE_DIR}"
  echo "config=${CONFIG}"
  echo "device=${DEVICE}"
  echo "total_updates=${TOTAL_UPDATES}"
  echo "save_interval=${SAVE_INTERVAL}"
  echo "seeds=${SEEDS}"
  echo "agent_encoders=${AGENT_ENCODERS}"
  echo "characters=${CHARACTERS}"
  echo "warmup_games=${WARMUP_GAMES}"
  echo "scored_games=${SCORED_GAMES}"
  echo "max_pairings=${MAX_PAIRINGS}"
  echo "learning_rate=${LEARNING_RATE}"
  echo "full_hyper_lr=${FULL_HYPER_LR}"
  echo "allow_train_failures=${ALLOW_TRAIN_FAILURES}"
} | tee "${BASE_DIR}/experiment_config.txt"

for seed in $SEEDS; do
  for pair in $AGENT_ENCODERS; do
    agent="${pair%%:*}"
    encoder="${pair##*:}"
    run_name="gpu_elo_${agent}_${encoder}_seed${seed}"
    run_path="${TRAIN_DIR}/${run_name}"
    final_checkpoint="${run_path}/checkpoints/update_$(printf '%04d' "$TOTAL_UPDATES").pt"
    if [[ -f "${run_path}/final_metrics.json" && -f "$final_checkpoint" ]]; then
      echo "skipping completed ${run_name}"
      continue
    fi
    lr_args=()
    if [[ "$agent" == "full_hyper" && -n "$FULL_HYPER_LR" ]]; then
      lr_args=(--learning-rate "$FULL_HYPER_LR")
    elif [[ -n "$LEARNING_RATE" ]]; then
      lr_args=(--learning-rate "$LEARNING_RATE")
    fi
    log_path="${LOG_DIR}/${run_name}.log"
    if [[ -e "$log_path" && ! -f "${run_path}/final_metrics.json" ]]; then
      log_path="${LOG_DIR}/${run_name}_retry_$(date +%Y%m%d_%H%M%S).log"
    fi
    echo "training ${run_name}"
    if ! ../.venv/bin/python -m latent_policy.ppo \
      --config "$CONFIG" \
      --agent "$agent" \
      --encoder "$encoder" \
      --seed "$seed" \
      --device "$DEVICE" \
      --total-updates "$TOTAL_UPDATES" \
      --save-interval "$SAVE_INTERVAL" \
      --run-dir "$TRAIN_DIR" \
      --run-name "$run_name" \
      "${lr_args[@]}" \
      --keep-checkpoints \
      --no-progress 2>&1 | tee "$log_path"; then
      echo "training failed for ${run_name}; see ${log_path}"
      if [[ "$ALLOW_TRAIN_FAILURES" != "1" ]]; then
        exit 1
      fi
    fi
  done
done

max_pairing_args=()
if [[ "$MAX_PAIRINGS" != "all" ]]; then
  max_pairing_args=(--max-pairings "$MAX_PAIRINGS")
fi

echo "running Elo tournament"
../.venv/bin/python -m latent_policy.melee_light_elo \
  --checkpoint-glob "${TRAIN_DIR}/*/checkpoints/update_*.pt" \
  --characters "$CHARACTERS" \
  --warmup-games "$WARMUP_GAMES" \
  --scored-games "$SCORED_GAMES" \
  "${max_pairing_args[@]}" \
  --min-pairings-per-competitor "$MIN_PAIRINGS_PER_COMPETITOR" \
  --elo-k "$ELO_K" \
  --episode-length 90 \
  --frame-skip 4 \
  --device "$DEVICE" \
  --seed "$ELO_SEED" \
  --output-dir "$ELO_DIR" \
  --progress-every "$PROGRESS_EVERY" 2>&1 | tee "${LOG_DIR}/elo.log"

echo "experiment complete: ${BASE_DIR}"
