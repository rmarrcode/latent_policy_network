# Experiment Notes

## 2026-04-27 Initial Implementation

Environment: `SwitchingDuelVecEnv`, a vectorized five-action cyclic dominance
game with hidden opponent types that switch mid-episode.

Policies tested:

- `static_mlp`
- `hyper_head`
- `film`
- `full_hyper`

### Quick Sweep

Command:

```bash
../.venv/bin/python -m latent_policy.sweep --config configs/quick.yaml \
  --agents static_mlp hyper_head film full_hyper --encoders gru \
  --seeds 1 --updates 12 --run-dir runs/quick_sweep --no-progress
```

Best quick run: `hyper_head + gru`, eval return `26.95`. `film + gru`
collapsed in the short run and should be treated as lower priority until tuned.

### Medium Sweep

Command:

```bash
../.venv/bin/python -m latent_policy.sweep --config configs/quick.yaml \
  --agents static_mlp hyper_head full_hyper --encoders mean gru attention \
  --seeds 1 2 --updates 40 --run-dir runs/medium_sweep --no-progress
```

Grouped means:

| agent | encoder | eval return | age 0-3 | age 4-15 | age 16+ | win rate |
|---|---:|---:|---:|---:|---:|---:|
| `full_hyper` | `gru` | 42.250 | 0.354 | 0.454 | 0.460 | 0.625 |
| `full_hyper` | `mean` | 41.765 | 0.330 | 0.448 | 0.462 | 0.617 |
| `hyper_head` | `attention` | 40.931 | 0.301 | 0.450 | 0.453 | 0.617 |
| `hyper_head` | `gru` | 40.839 | 0.336 | 0.452 | 0.435 | 0.617 |
| `full_hyper` | `attention` | 40.548 | 0.325 | 0.441 | 0.442 | 0.611 |
| `hyper_head` | `mean` | 35.867 | 0.337 | 0.395 | 0.368 | 0.591 |
| `static_mlp` | `mean` | 33.273 | 0.316 | 0.353 | 0.353 | 0.575 |

Takeaway: generated actor weights are useful in this environment. The strongest
current default is `full_hyper + gru`; `hyper_head + gru` is a cheaper runner-up.

### Same-Budget Best Config Runs

Command pattern:

```bash
../.venv/bin/python -m latent_policy.ppo --config configs/best.yaml \
  --agent AGENT --encoder ENCODER --run-name NAME --no-progress
../.venv/bin/latent-policy-eval runs/best/NAME/checkpoint.pt --episodes 256
```

All runs used 120 updates, 96 vectorized envs, 128-step rollouts, and 256
deterministic eval episodes.

| agent | encoder | eval return | age 0-3 | age 4-15 | age 16+ | win rate |
|---|---:|---:|---:|---:|---:|---:|
| `full_hyper` | `gru` | 66.887 | 0.356 | 0.538 | 0.560 | 0.687 |
| `hyper_head` | `gru` | 62.551 | 0.336 | 0.509 | 0.517 | 0.663 |
| `static_mlp` | `mean` | 58.828 | 0.351 | 0.473 | 0.482 | 0.647 |

The full generated actor improves by about `+8.06` return over the static MLP
under the same training budget. Most of the gain appears after a few interaction
steps with the new opponent rather than in the first four post-switch steps.

## 2026-04-28 Public Environment Expansion

Added adapters in `latent_policy/public_envs.py` and a suite runner in
`latent_policy/public_suite.py`.

Installed and ran:

- OpenSpiel: `matrix_rps`, `matrix_rpsw`, `matrix_pd`, `kuhn_poker`,
  `tic_tac_toe`, `connect_four`
- PettingZoo Classic: `rps_v2`
- MPE2: `simple_adversary_v3`, `simple_push_v3`, `simple_tag_v3`
- SlimeVolleyGym: `SlimeVolley-v0`
- MAgent2: `battle_v4`
- FootsiesGym: headless Footsies binary

Dependency blockers:

- Melting Pot: `dm-meltingpot` could not install because `dmlab2d` has no
  matching distribution in this environment.
- Google Research Football: `gfootball` source build failed because SDL2 CMake
  development files are missing (`SDL2Config.cmake` / `sdl2-config.cmake`).
- Overcooked-AI imports and can instantiate its base MDP/env, but I did not add
  a training adapter in this pass because it needs a cooperative two-agent
  featurization/action wrapper rather than the current opponent-policy wrapper.

### Public Suite

Command:

```bash
../.venv/bin/python -m latent_policy.public_suite \
  --updates 8 --agents static_mlp hyper_head full_hyper \
  --seeds 1 --run-dir runs/public_suite --no-progress
```

All 33 runs completed without errors. Best quick result per env:

| env | best agent | eval return | note |
|---|---:|---:|---|
| `openspiel_matrix_rps` | `hyper_head` | 14.984 | adaptive head helped |
| `openspiel_matrix_rpsw` | `hyper_head` | 23.438 | adaptive head helped |
| `openspiel_matrix_pd` | tie | 371.688 | payoff scale dominates; not diagnostic |
| `openspiel_kuhn` | `static_mlp` | 0.688 | short game, little adaptation signal |
| `openspiel_tictactoe` | `static_mlp` | 0.797 | action masking absent, static did best |
| `openspiel_connect_four` | `full_hyper` | 1.000 | very small budget |
| `pettingzoo_rps` | `hyper_head` | 17.797 | small edge over static |
| `mpe_simple_adversary` | `full_hyper` | -61.397 | all poor, but full-hyper least bad |
| `mpe_simple_push` | `full_hyper` | 46.372 | generated actor helped in quick run |
| `mpe_simple_tag` | `hyper_head` | 7.188 | small positive score |
| `slime_volley` | `static_mlp` | -2.094 | no useful learning at this budget |

### Medium Public Follow-Up

Command:

```bash
../.venv/bin/python -m latent_policy.public_suite \
  --envs openspiel_matrix_rps openspiel_matrix_rpsw pettingzoo_rps \
         mpe_simple_push openspiel_kuhn \
  --agents static_mlp hyper_head full_hyper \
  --updates 24 --seeds 1 2 --run-dir runs/public_medium --no-progress

../.venv/bin/python -m latent_policy.public_suite \
  --envs openspiel_matrix_rps openspiel_matrix_rpsw pettingzoo_rps \
         mpe_simple_push openspiel_kuhn \
  --agents film --updates 24 --seeds 1 2 \
  --run-dir runs/public_medium_film --no-progress
```

Grouped mean eval returns:

| env | static | hyper_head | full_hyper | film |
|---|---:|---:|---:|---:|
| `openspiel_matrix_rps` | 12.594 | 26.742 | 19.891 | -0.078 |
| `openspiel_matrix_rpsw` | 22.156 | 23.750 | 22.492 | -0.094 |
| `pettingzoo_rps` | 16.875 | 16.875 | 16.875 | 3.734 |
| `mpe_simple_push` | 65.551 | 62.653 | 49.924 | -3.721 |
| `openspiel_kuhn` | 0.289 | -0.352 | -0.297 | 0.164 |

Takeaways:

- The clearest public-environment win for generated weights is OpenSpiel
  repeated `matrix_rps`: `hyper_head` more than doubled static under the same
  24-update, two-seed budget.
- `full_hyper` did not dominate public tasks the way it did in the synthetic
  switching duel. Its larger generated actor appears to need more data or
  better regularization on these smaller/noisier public tasks.
- FiLM was consistently weak in these settings.
- Turn-based games without action masking are not yet a fair test for the
  generated-weight idea; adding action masks should be the next code change
  before interpreting TicTacToe/ConnectFour seriously.

### Heavy Public Probes

Footsies:

```bash
../.venv/bin/python -m latent_policy.public_suite \
  --envs footsies --agents static_mlp hyper_head full_hyper \
  --updates 2 --seeds 1 --run-dir runs/public_footsies --no-progress
```

Tiny-budget returns: `static_mlp 0.125`, `hyper_head -0.250`,
`full_hyper 0.000`. This mainly validates the headless binary adapter.

MAgent2 Battle:

```bash
../.venv/bin/python -m latent_policy.public_suite \
  --envs magent_battle --agents static_mlp hyper_head full_hyper \
  --updates 2 --seeds 1 --run-dir runs/public_magent --no-progress
```

Tiny-budget returns: `static_mlp -1.360`, `hyper_head -2.248`,
`full_hyper -0.010`. This validates the many-agent PettingZoo-style adapter.

## 2026-04-30 Melee Light Training Pass

Added dedicated configs:

- `configs/melee_light_lvl0.yaml`
- `configs/melee_light_lvl3.yaml`

These use the built-in CPU opponent on the default Fox-vs-Marth setup with
`frame_skip: 4`, `max_episode_frames: 240`, and 60-step training episodes.
This is important context: unlike `SwitchingDuelVecEnv`, the current
`melee_light_knockback` adapter is not a hidden-opponent-switching benchmark,
so it is a much weaker test of the latent-policy idea.

### Broad Sweep

Training pattern:

```bash
for level in 0 3; do
  for seed in 1 2; do
    for agent in static_mlp hyper_head full_hyper film; do
      encoder=gru
      if [ "$agent" = "static_mlp" ]; then
        encoder=mean
      fi
      ../.venv/bin/python -m latent_policy.ppo \
        --config "configs/melee_light_lvl${level}.yaml" \
        --agent "$agent" \
        --encoder "$encoder" \
        --seed "$seed" \
        --total-updates 16 \
        --run-dir runs/melee_light_apr30_train \
        --run-name "melee_light_lvl${level}_${agent}_${encoder}_seed${seed}" \
        --no-progress
    done
  done
done
```

Deterministic evaluation: 32 episodes per checkpoint. Aggregated outputs:

- `runs/melee_light_apr30_eval_lvl0.csv`
- `runs/melee_light_apr30_eval_lvl3.csv`
- `runs/melee_light_apr30_grouped.csv`

Grouped means:

| level | agent | encoder | eval return | win rate | loss rate | eval length |
|---|---:|---:|---:|---:|---:|---:|
| `0` | `film` | `gru` | `-1.000` | `0.000` | `1.000` | `59.875` |
| `0` | `full_hyper` | `gru` | `-0.625` | `0.188` | `0.812` | `51.969` |
| `0` | `hyper_head` | `gru` | `-1.000` | `0.000` | `1.000` | `59.922` |
| `0` | `static_mlp` | `mean` | `-1.000` | `0.000` | `1.000` | `59.922` |
| `3` | `film` | `gru` | `-0.344` | `0.328` | `0.672` | `18.156` |
| `3` | `full_hyper` | `gru` | `-1.000` | `0.000` | `1.000` | `24.500` |
| `3` | `hyper_head` | `gru` | `-1.000` | `0.000` | `1.000` | `35.922` |
| `3` | `static_mlp` | `mean` | `-1.000` | `0.000` | `1.000` | `59.922` |

The only non-trivial short-run positives were:

- `full_hyper` at level `0`, seed `2`: return `-0.25`, win rate `0.375`
- `film` at level `3`, seed `2`: return `0.3125`, win rate `0.65625`

Those were enough to justify a longer follow-up, but not enough to claim a
stable edge on their own because they came from single seeds out of 32 eval
episodes.

### Longer Follow-Up

I extended the only two promising pairs:

- level `0`: `static_mlp` vs `full_hyper`
- level `3`: `static_mlp` vs `film`

Training pattern:

```bash
../.venv/bin/python -m latent_policy.ppo \
  --config configs/melee_light_lvl0.yaml \
  --agent static_mlp --encoder mean --seed 1 --total-updates 32 \
  --run-dir runs/melee_light_apr30_followup \
  --run-name melee_light_lvl0_long_static_mlp_mean_seed1 --no-progress

../.venv/bin/python -m latent_policy.ppo \
  --config configs/melee_light_lvl0.yaml \
  --agent full_hyper --encoder gru --seed 1 --total-updates 32 \
  --run-dir runs/melee_light_apr30_followup \
  --run-name melee_light_lvl0_long_full_hyper_gru_seed1 --no-progress

../.venv/bin/python -m latent_policy.ppo \
  --config configs/melee_light_lvl3.yaml \
  --agent static_mlp --encoder mean --seed 1 --total-updates 32 \
  --run-dir runs/melee_light_apr30_followup \
  --run-name melee_light_lvl3_long_static_mlp_mean_seed1 --no-progress

../.venv/bin/python -m latent_policy.ppo \
  --config configs/melee_light_lvl3.yaml \
  --agent film --encoder gru --seed 1 --total-updates 32 \
  --run-dir runs/melee_light_apr30_followup \
  --run-name melee_light_lvl3_long_film_gru_seed1 --no-progress
```

I repeated the same pattern for seed `2`, then evaluated each checkpoint over
32 deterministic episodes. Aggregated outputs:

- `runs/melee_light_apr30_followup_eval_lvl0.csv`
- `runs/melee_light_apr30_followup_eval_lvl3.csv`

Grouped means:

| level | agent | encoder | eval return | win rate | loss rate | eval length |
|---|---:|---:|---:|---:|---:|---:|
| `0` | `full_hyper` | `gru` | `-1.000` | `0.000` | `1.000` | `59.688` |
| `0` | `static_mlp` | `mean` | `-1.000` | `0.000` | `1.000` | `59.922` |
| `3` | `film` | `gru` | `-1.000` | `0.000` | `1.000` | `41.625` |
| `3` | `static_mlp` | `mean` | `-0.156` | `0.422` | `0.578` | `30.969` |

### Melee Light Takeaway

- I did not find a reliable Melee Light advantage for the latent-policy
  architectures.
- The broad sweep produced two isolated positive seeds, but neither survived the
  longer 32-update follow-up.
- In the longer level-`3` follow-up, the static MLP actually produced the best
  single run: return `0.6875`, win rate `0.84375`.
- The current Melee Light adapter appears to be a poor fit for the latent-policy
  hypothesis because it uses a fixed built-in CPU opponent rather than the
  hidden-opponent-switching structure that generated-weight adaptation is meant
  to exploit.

## 2026-05-01 Melee Light Self-Play Adapter

Changed Melee Light so P2 can be controlled externally instead of by the built-in
CPU:

- `latent_policy/melee_light_runtime/runtime_bridge.js` now accepts an optional
  P2 action in `env.step(action, opponentAction, callback)`.
- `MeleeLightKnockbackEnv.step()` now accepts `opponent_action`.
- `GymSingleDiscreteVecEnv` now detects Melee Light external-opponent mode and
  drives P2 with hidden scripted opponent policies.
- `configs/melee_light_lvl0.yaml`, `configs/melee_light_lvl3.yaml`, and the
  `public_suite` Melee Light spec now default to `opponent_control: external`.

Default opponent style pool:

| style | behavior |
|---|---|
| `rushdown` | closes distance, jumps occasionally, and attacks when close |
| `approach_jab` | moves in and uses jabs/tilts |
| `spacer` | backs off when too close and pokes at mid-range |
| `zoner` | retreats and uses specials from distance |
| `counter_poke` | shields at close range and pokes otherwise |
| `jumper` | jump-heavy vertical pressure |
| `mirror_agent` | mirrors the learner's current action with left/right inversion |
| `anti_frequency` | responds to the learner's most common action class |

Default character pools:

| side | characters |
|---|---|
| learner | Fox, Falco, Falcon |
| scripted opponent | Marth, Puff, Fox, Falco, Falcon |

This makes Melee Light a better fit for latent policies than the old CPU setup:
the agent must infer both the opponent's style and matchup from recent game
state, and the style can switch mid-episode through `switch_hazard`.

Verification:

```bash
../.venv/bin/pytest -q tests/test_melee_light_env.py tests/test_public_envs.py
```

Browser smoke:

```bash
../.venv/bin/python - <<'PY'
from latent_policy.melee_light_env import MeleeLightKnockbackEnv

env = MeleeLightKnockbackEnv(
    frame_skip=4,
    max_episode_frames=32,
    opponent_control="external",
    agent_character=2,
    opponent_character=3,
    opponent_level=0,
)
try:
    obs, info = env.reset(seed=123)
    for action, opponent_action in [(2, 1), (7, 6), (15, 14)]:
        obs, reward, terminated, truncated, info = env.step(action, opponent_action=opponent_action)
        if terminated or truncated:
            break
finally:
    env.close()
PY
```

## 2026-05-01 Melee Light Self-Play-Only Experiment

I ran another Melee Light pass using only the external P2 path:
`configs/melee_light_lvl0.yaml` has `opponent_control: external`, and no
built-in CPU opponent was used. P2 was controlled by the hidden scripted
self-play-style pool from the adapter above. This is still not a separately
learned opponent policy, but it gives the learner an opponent whose strategy and
character are hidden and switchable.

### Broad 12-Update Sweep

Training pattern:

```bash
for seed in 1 2; do
  for pair in static_mlp:mean hyper_head:gru full_hyper:gru film:gru; do
    agent="${pair%%:*}"
    encoder="${pair##*:}"
    ../.venv/bin/python -m latent_policy.ppo \
      --config configs/melee_light_lvl0.yaml \
      --agent "$agent" \
      --encoder "$encoder" \
      --seed "$seed" \
      --total-updates 12 \
      --run-dir runs/melee_light_selfplay_may01 \
      --run-name "selfplay_${agent}_${encoder}_seed${seed}" \
      --no-progress
  done
done
```

Evaluation: 32 deterministic episodes per checkpoint.

```bash
../.venv/bin/python -m latent_policy.evaluate \
  runs/melee_light_selfplay_may01/selfplay_full_hyper_gru_seed1/checkpoint.pt \
  --episodes 32
```

I repeated evaluation for all checkpoints from both seeds and combined the
outputs into:

- `runs/melee_light_selfplay_may01_eval_seed1.csv`
- `runs/melee_light_selfplay_may01_eval_seed2.csv`
- `runs/melee_light_selfplay_may01_all_eval.csv`
- `runs/melee_light_selfplay_may01_grouped.csv`

Grouped means:

| phase | agent | encoder | seeds | eval return | win rate | loss rate | eval length | positive step rate | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `selfplay_12_update` | `full_hyper` | `gru` | `2` | `-0.688` | `0.156` | `0.844` | `25.109` | `0.0068` | `24.0` |
| `selfplay_12_update` | `static_mlp` | `mean` | `2` | `-0.719` | `0.141` | `0.859` | `29.578` | `0.0055` | `32.5` |
| `selfplay_12_update` | `hyper_head` | `gru` | `2` | `-0.844` | `0.078` | `0.922` | `33.453` | `0.0025` | `36.0` |
| `selfplay_12_update` | `film` | `gru` | `2` | `-0.906` | `0.047` | `0.953` | `36.328` | `0.0017` | `35.5` |

Per-run result:

| agent | seed | eval return | win rate | loss rate | eval length | switches |
|---|---:|---:|---:|---:|---:|---:|
| `static_mlp` | `1` | `-0.438` | `0.281` | `0.719` | `25.406` | `24` |
| `full_hyper` | `1` | `-0.812` | `0.094` | `0.906` | `29.250` | `29` |
| `hyper_head` | `1` | `-0.750` | `0.125` | `0.875` | `29.625` | `32` |
| `film` | `1` | `-1.000` | `0.000` | `1.000` | `45.125` | `43` |
| `static_mlp` | `2` | `-1.000` | `0.000` | `1.000` | `33.750` | `41` |
| `full_hyper` | `2` | `-0.562` | `0.219` | `0.781` | `20.969` | `19` |
| `hyper_head` | `2` | `-0.938` | `0.031` | `0.969` | `37.281` | `40` |
| `film` | `2` | `-0.812` | `0.094` | `0.906` | `27.531` | `28` |

The 12-update sweep gave `full_hyper + gru` the best mean return, but the edge
over static was small: `+0.031` return and `+0.016` win rate. Static also had
the best single seed, so this pass alone was not enough to claim a stable
latent advantage.

### 24-Update Follow-Up

I extended only the strongest latent candidate and the static baseline:

```bash
for seed in 1 2; do
  for pair in static_mlp:mean full_hyper:gru; do
    agent="${pair%%:*}"
    encoder="${pair##*:}"
    ../.venv/bin/python -m latent_policy.ppo \
      --config configs/melee_light_lvl0.yaml \
      --agent "$agent" \
      --encoder "$encoder" \
      --seed "$seed" \
      --total-updates 24 \
      --run-dir runs/melee_light_selfplay_may01_followup \
      --run-name "selfplay_long_${agent}_${encoder}_seed${seed}" \
      --no-progress
  done
done
```

Grouped means:

| phase | agent | encoder | seeds | eval return | win rate | loss rate | eval length | positive step rate | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `selfplay_24_update` | `full_hyper` | `gru` | `2` | `-0.406` | `0.297` | `0.703` | `32.406` | `0.0127` | `30.5` |
| `selfplay_24_update` | `static_mlp` | `mean` | `2` | `-0.625` | `0.188` | `0.812` | `38.609` | `0.0048` | `40.5` |

Per-run result:

| agent | seed | eval return | win rate | loss rate | eval length | switches |
|---|---:|---:|---:|---:|---:|---:|
| `full_hyper` | `1` | `-0.688` | `0.156` | `0.844` | `44.906` | `44` |
| `static_mlp` | `1` | `-0.688` | `0.156` | `0.844` | `35.469` | `33` |
| `full_hyper` | `2` | `-0.125` | `0.438` | `0.562` | `19.906` | `17` |
| `static_mlp` | `2` | `-0.562` | `0.219` | `0.781` | `41.750` | `48` |

### Self-Play Takeaway

- The external self-play-style Melee Light setup is a better match for the
  latent-policy hypothesis than the old CPU setup because opponent style and
  character are hidden variables that must be inferred online.
- `full_hyper + gru` is the best current Melee Light self-play agent. In the
  24-update follow-up it beat static by `+0.219` return and `+0.109` win rate.
- `hyper_head + gru` and `film + gru` did not show an advantage in this run.
- The conclusion is still bounded by a small sample: two seeds and 32 eval
  episodes per checkpoint. The result is a useful edge, not a final benchmark.

## 2026-05-01 Long GPU Melee Light Elo Experiment

I added an extensive GPU experiment path for policy-vs-policy Melee Light Elo:

- `configs/melee_light_gpu_elo.yaml` trains on all five available characters
  with `opponent_control: external`, `context_len: 32`, 90-step episodes, and
  update-numbered checkpoint retention.
- `latent_policy.ppo` now supports `keep_checkpoints` plus
  `--keep-checkpoints` / `--save-interval`, so Elo can evaluate points in
  training instead of only the final checkpoint.
- `latent_policy.melee_light_elo` treats each
  `agent x checkpoint_update x character` item as an Elo competitor.
- `scripts/run_melee_light_gpu_elo_experiment.sh` runs the full train/eval
  sequence.

Launched run:

```bash
tmux new-session -d -s melee_elo_extensive \
  "cd /home/ryan-marr/Documents/secret/latent_policy_env/latent_policy && \
   RUN_TAG=melee_light_gpu_elo_may01_extensive \
   TOTAL_UPDATES=120 SAVE_INTERVAL=30 SEEDS='1 2' \
   AGENT_ENCODERS='static_mlp:mean hyper_head:gru full_hyper:gru film:gru' \
   CHARACTERS='marth,puff,fox,falco,falcon' \
   WARMUP_GAMES=16 SCORED_GAMES=48 MAX_PAIRINGS=320 \
   MIN_PAIRINGS_PER_COMPETITOR=4 PROGRESS_EVERY=5 DEVICE=cuda \
   bash scripts/run_melee_light_gpu_elo_experiment.sh 2>&1 | \
   tee runs/melee_light_gpu_elo_may01_extensive/experiment.log"
```

Protocol:

- Training: 4 architectures x 2 seeds x 120 updates, saving update `30`, `60`,
  `90`, and `120`.
- Stability adjustment: the first `full_hyper + gru` seed-`1` attempt at the
  default `3e-4` learning rate produced NaN logits after update `30`, so I
  moved that partial run under `runs/melee_light_gpu_elo_may01_extensive/failed/`
  and resumed `full_hyper` training at `1e-4`.
- Competitors: checkpoint points crossed with Marth, Puff, Fox, Falco, and
  Falcon.
- Elo pairings: 320 sampled pairings, both side orders per pairing.
- Adaptation window: each side-order series has 16 warmup games followed by 48
  scored games. Policy context is reset only at the start of the side-order
  series and is preserved across games inside the series.

Expected outputs:

- `runs/melee_light_gpu_elo_may01_extensive/train/`
- `runs/melee_light_gpu_elo_may01_extensive/elo/elo.csv`
- `runs/melee_light_gpu_elo_may01_extensive/elo/pair_results.csv`
- `runs/melee_light_gpu_elo_may01_extensive/elo/agent_update_character_summary.csv`
- `runs/melee_light_gpu_elo_may01_extensive/elo/report.md`
