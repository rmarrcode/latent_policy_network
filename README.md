# Latent Policy

This project is a compact research scaffold for generated-weight RL policies in
rapidly changing 1v1 settings. The environment is a vectorized repeated duel
where the opponent has a hidden style that can switch mid-episode. The adaptive
agents encode recent interaction context and generate policy weights on the fly.

## Setup

From `latent_policy_env`:

```bash
source .venv/bin/activate
cd latent_policy
pip install -r requirements.txt
pip install -e .
```

The venv in this workspace already has the required packages installed.

For the public environment adapters:

```bash
pip install -r requirements-public.txt
pip install -e .
```

## Train One Run

```bash
../.venv/bin/python -m latent_policy.ppo --config configs/quick.yaml --agent hyper_head --encoder gru
```

Outputs are written under `latent_policy/runs/...`:

- `metrics.csv`: training and eval metrics per update
- `checkpoint.pt`: latest checkpoint
- `final_metrics.json`: final run summary

## Architecture Sweep

```bash
./scripts/run_quick_sweep.sh
```

Summarize any sweep:

```bash
../.venv/bin/python -m latent_policy.analyze runs/medium_sweep/summary.csv
```

The sweep compares:

- `static_mlp`: regular non-adaptive actor-critic baseline
- `hyper_head`: context network generates the actor's final linear head
- `film`: context network modulates actor features with FiLM parameters
- `full_hyper`: context network generates every actor-layer weight

Context encoders:

- `mean`: average recent observations, then MLP
- `flat`: flatten recent observations, then MLP
- `gru`: recurrent context encoder
- `attention`: small Transformer encoder

## Overnight Run

```bash
./scripts/run_overnight_sweep.sh
```

This runs three seeds across static and generated-weight policies using GRU and
attention encoders. It writes `runs/overnight/summary.csv` as runs complete.

## Metrics To Watch

- `eval_return_mean`: average episode return under deterministic policy
- `eval_reward_age_0_3`: performance immediately after an opponent switch
- `eval_reward_age_4_15`: early adaptation window
- `eval_reward_age_16_plus`: settled performance after inference
- `rollout_return_recent`: recent training episode return

The most relevant research signal is whether adaptive policies beat
`static_mlp`, especially in `age_0_3` and `age_4_15` after switches.

## Public Environments

Run the public environment suite:

```bash
../.venv/bin/python -m latent_policy.public_suite --updates 8 --no-progress
```

The suite currently covers OpenSpiel matrix/turn games, PettingZoo RPS, MPE2
tasks, SlimeVolley, MAgent2 Battle, and Footsies. Footsies is much slower
because it launches headless game-server processes.

## Melee Light

I added dedicated Melee Light configs:

- `configs/melee_light_lvl0.yaml`
- `configs/melee_light_lvl3.yaml`

These runs use the built-in CPU opponent on the default Fox-vs-Marth setup with
`frame_skip: 4` and 60-step episodes. Unlike the synthetic switching-duel
benchmark, this adapter does not currently create a hidden opponent-type switch
inside the episode, so the adaptation signal is much weaker.

I ran a broad 16-update sweep across `static_mlp`, `hyper_head`, `full_hyper`,
and `film`, using two seeds and 32 deterministic eval episodes per checkpoint:

| level | best agent | eval return | win rate |
|---|---:|---:|---:|
| `0` | `full_hyper` | `-0.625` | `0.188` |
| `3` | `film` | `-0.344` | `0.328` |

Those short-run positives did not hold up in a longer 32-update follow-up:

| level | baseline | latent candidate |
|---|---:|---:|
| `0` | `static_mlp -1.000 / 0.000 win` | `full_hyper -1.000 / 0.000 win` |
| `3` | `static_mlp -0.156 / 0.422 win` | `film -1.000 / 0.000 win` |

Takeaway: I did not find a reliable Melee Light edge for the latent-policy
architectures. The combined result files are `runs/melee_light_apr30_all_eval.csv`
and `runs/melee_light_apr30_grouped.csv`.

## Current Local Results

I ran a two-seed medium sweep with 40 updates per run. Mean eval return:

| agent | encoder | return |
|---|---:|---:|
| `full_hyper` | `gru` | 42.250 |
| `full_hyper` | `mean` | 41.765 |
| `hyper_head` | `attention` | 40.931 |
| `hyper_head` | `gru` | 40.839 |
| `static_mlp` | `mean` | 33.273 |

Based on that pass, `configs/best.yaml` and `configs/overnight.yaml` default
to `full_hyper + gru`.

I also ran same-budget 120-update runs with `configs/best.yaml` and evaluated
each checkpoint over 256 episodes:

| agent | encoder | return | win rate |
|---|---:|---:|---:|
| `full_hyper` | `gru` | 66.887 | 0.687 |
| `hyper_head` | `gru` | 62.551 | 0.663 |
| `static_mlp` | `mean` | 58.828 | 0.647 |
