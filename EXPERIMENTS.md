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
