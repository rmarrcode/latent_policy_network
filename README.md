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
- `context_mlp`: context encoder followed by a regular MLP actor-critic with fixed learned weights
- `hyper_head`: context network generates the actor's final linear head
- `film`: context network modulates actor features with FiLM parameters
- `full_hyper`: context network generates every actor-layer weight

Context encoders:

- `mean`: average recent observations, then MLP
- `flat`: flatten recent observations, then MLP
- `gru`: recurrent context encoder
- `attention`: small Transformer encoder

## Parameter-Matched Comparison

Compare the latent architecture in a config with a `static_mlp` baseline whose
hidden width is automatically chosen to minimize the parameter-count difference:

```bash
../.venv/bin/python -m latent_policy.compare --config configs/best.yaml --dry-run
../.venv/bin/python -m latent_policy.compare --config configs/best.yaml --seeds 1 2 3 --no-progress
```

For a shorter comparison, use `--config configs/quick.yaml --updates 12`.
Override the latent architecture with `--latent-agent full_hyper` (or
`hyper_head` / `film`) and its encoder with `--encoder gru`.

Matching counts all trainable parameters, including the context encoder,
generator, and critic. Generated actor weights are outputs, so they are not
counted again. The latent architecture stays at its configured size; only the
baseline hidden width changes. The script rejects pairs differing by more than
1% of the latent model's count; adjust this with `--param-tolerance 0.02` for 2%.

Both models use the same environment configuration, seeds, PPO settings,
training steps, and evaluation settings. Final evaluation is always enabled,
even if the input config disables it. This matches parameter and environment-step
budgets; runtime and PPO early stopping can still differ between architectures.
The default uses three seeds and prints per-seed return differences and mean
returns across seeds. Outputs in `runs/parameter_matched` include
`comparison_plan.json`, `summary.csv` with counts and evaluation metrics, and
the usual per-run training outputs. Use `--run-dir` to keep experiments separate.
`--dry-run` saves the comparison plan without training (environment discovery
still requires the configured environment's dependencies).

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

## Real Melee Self-Play

For the controlled five-model comparison, use:

```bash
bash scripts/run_melee_comparison.sh --config configs/melee_fair_self_play.yaml
```

This config runs three training seeds and compares:

| Model | Parameters | Actor/trunk width | History |
|---|---:|---:|---|
| Static MLP | 130,167 | 332 | None; memory ablation |
| GRU + regular MLP (`context_mlp`) | 129,955 | 270 | Same 32 observations |
| FiLM | 130,011 | 118 | Same 32 observations |
| Hyper-head | 128,975 | 62 | Same 32 observations |
| Full hypernetwork | 130,247 | 64 | Same 32 observations |

The primary regular-policy comparison is **GRU + MLP versus the generated-weight
models**. All four history-aware policies use the same 64-unit GRU and 64-dimensional
context representation, initialized identically within a seed but learned
independently. Actor/trunk widths are selected automatically to match the full
hypernetwork's total trainable parameter count within 1%. The static MLP is an
additional memory ablation, not sufficient evidence for a benefit from generating
weights. These controls match total capacity and information, not the allocation
of parameters between actors and critics or the exact computation performed.

Every model receives the same PPO decision budget, optimizer settings, seeds,
and a common frozen opponent pool. Update order rotates across models. The pool
is updated only after every model finishes the current update. The learner's
history survives an unannounced opponent change; only rollout boundaries clear
it. Controller port stays fixed within a rollout. No opponent identity or switch
flag is supplied as policy input.

Additional outputs include `inference_cost.csv` (warm batch-one action latency,
with CUDA synchronization), rollout/optimization/cumulative training wall times
in `training.csv`, and `training_bouts.csv`. The training metrics include rewards
in decision-age windows 0–3, 4–15, and 16+ after a switch, plus returns from the
first, second, and later bouts. Counts accompany each mean; sparse first-hit
rewards make these decision-age windows a supplementary diagnostic.

The fair config bounds Elo evaluation to a reproducible connected graph with
eight opponents per saved checkpoint, equal coverage, and balanced controller
ports. With 75 checkpoints this is 300 pairings rather than 2,775. The graph is
chosen before matches; ratings share one fitted scale, but a checkpoint's exact
opponents differ. Set `eval_opponents_per_checkpoint: 0` for full round-robin
evaluation. `evaluation_plan.json` records coverage and the evaluation seed.

After Elo evaluation, a separate switch test evaluates each final model against
every architecture's final checkpoint from the next training seed. Those opponent
weights were never in that learner's training pool. Both controller ports are
tested. The learner first encounters a common initial reference opponent for
two bouts, then the opponent is replaced without clearing the learner's context.
Four scored bouts track immediate and later performance. Results go to
`switch_bouts.csv`, `switch_summary.csv`, `switch_protocol.json`, and
`switch_adaptation.png`. With only one training seed this test falls back to
same-seed opponents and explicitly records that it is not a held-out-seed test.

A supported conclusion would be that context-conditioned weight generation
improves performance over a history- and parameter-matched regular policy under
these opponent-switch conditions. It would not establish that latent policies
are universally better. Assess all three seeds, inference/runtime costs, draw
rates, held-out-seed performance, and Elo sampling uncertainty before claiming an
advantage. The graph's across-seed spread is not a confidence interval.

The Melee Light browser environment has been removed. The default is now
[Slippi Dolphin + libmelee](https://github.com/vladfi1/libmelee), using the real
Melee engine. This is the evaluation target for eventual comparisons with
[Phillip II / Slippi-AI](https://github.com/vladfi1/slippi-ai), which uses the
same Dolphin/libmelee state and controller interfaces. An external Phillip II
opponent adapter is not yet implemented. When adding that comparison, record the
model version and match character, stage, controller-port order, input delay,
and warmup rules. First-hit scores are a custom benchmark and are not comparable
to published full-match ratings.

The optional native backend is
[melee-sim-light](https://github.com/kyhavlov/melee-sim-light), a native C simulator
built from the Melee decompilation with a Python API. It steps and resets directly,
without Dolphin or menus. The simulator still requires extracted game data from
a Melee NTSC 1.02 ISO; the decompiled source does not include that data. Its
upstream fidelity claims have not been independently verified by this project.

Build/install the pinned simulator, optionally extract an ISO, and train:

```bash
bash scripts/setup_melee_decomp.sh /path/to/Melee-NTSC-1.02.iso data
MSL_DATA_DIR="$PWD/data" bash scripts/run_melee_comparison.sh --backend decomp --run-dir runs/melee_decomp
```

If the data is already extracted, installation needs no ISO:

```bash
bash scripts/setup_melee_decomp.sh
bash scripts/run_melee_comparison.sh --backend decomp --data-dir /path/to/extracted/data --dry-run
bash scripts/run_melee_comparison.sh --backend decomp --data-dir /path/to/extracted/data --run-dir runs/melee_decomp
```

The same first-hit rules, observations, action mapping, parameter matching,
self-play trainer, and Elo evaluator are used by both backends. The current
native adapter uses one match at a time; it does not yet exploit the simulator's
batched throughput. Native reset seeds are deterministic per training seed.
The adapter's mapping and first-hit handling are tested with a test backend;
live gameplay validation requires extracted data. No training run should be
reported as started merely because the simulator builds or a dry run passes.

The default backend runs Melee through
[libmelee](https://github.com/vladfi1/libmelee) and its documented headless ExiAI
Dolphin build. To use that backend, install the Python adapter and pinned Linux
x86-64 Dolphin build:

```bash
bash scripts/setup_melee.sh
export MELEE_ISO="/absolute/path/to/your/Melee-NTSC-1.02.iso"
bash scripts/run_melee_comparison.sh --dry-run
bash scripts/run_melee_comparison.sh --backend slippi --run-dir runs/melee_first_hit
```

The setup script downloads the build linked by libmelee's maintainers, checks its
SHA-256, and checks its version. It does not download game data. You must provide
your Melee NTSC 1.02 ISO. The setup script extracts the AppImage once and uses
`squashfs-root/AppRun`, so FUSE is not required and version parsing is unaffected
by AppImage extraction messages. A unique complete NTSC 1.02 image in `~/Downloads`
is discovered automatically; `MELEE_ISO` or `--iso` overrides discovery. Use
`--dolphin` for an explicit executable path. The default config
is `configs/melee_self_play.yaml`; `--device cpu`, `--updates 10`, and `--seeds 1`
can shorten an experiment. `--dry-run` checks model sizes without starting Dolphin
or requiring an ISO. Other platforms need their own compatible Slippi executable
and may need `headless: false` in the config.

Rules and observations:

- Both controller ports are controlled by policies; no built-in CPU is used.
- The first frame with increased damage ends the bout, with rewards +1 / -1.
  Shield damage does not count. Same-frame trades, no-hit stock losses, and
  timeouts are draws. Hits are checked every emulated frame, even within an
  action repeat. Missing frames or emulator failures stop the experiment.
- A fresh match starts every bout with zero damage. The native backend resets
  directly. Slippi relaunches its isolated Dolphin process and selects both
  characters and the stage. This avoids a verified pause/frame-stream hang when
  trying to quit a bout with L+R+A+Start in the ExiAI build. The relaunch takes
  about 2.2 seconds on this machine, adding substantial overhead to short bouts.
  Menu/countdown frames produce no training samples.
- Defaults are Fox dittos on Final Destination, four frames per decision, and
  a 600-frame (10 game-second) bout limit. The 22 discrete actions cover movement,
  jump, shield, grab, attacks, specials, and C-stick directions. The 35 features
  describe own/opponent fighter state, relative position, and bout progress.
- The Slippi backend has been checked with the NTSC 1.02 image: repeated fresh
  resets and a controlled first-hit scenario return the expected zero-sum
  rewards. This workflow uses a small action/state interface; it does not
  include a pretrained Melee controller.

The comparison automatically widens `static_mlp` to match `full_hyper + gru`
within 1% of total trainable parameters, including encoders, generators, and
critics. Defaults are **130,167 baseline vs 130,247 latent parameters (0.061%)**.
Change the latent architecture in `policy` and the tolerance in
`self_play.param_tolerance`.

Each update gives both learners the same number of environment decisions and
PPO optimization epochs against a common frozen pool containing recent snapshots
of both architectures. The pool changes only after both learners update. Controller
ports are sampled during training. Opponents remain fixed for four bouts so the
history-aware models can carry context across bouts and opponent switches;
contexts reset at rollout boundaries. The trainer always runs all PPO epochs
for equal optimization budgets; `target_kl` must be null. All three seeds use the
same hyperparameters. Defaults use 512 decisions per rollout so even no-hit
rollouts can complete bouts before the rollout boundary. Completed bout counts,
frames, and runtime can differ
because policies end bouts at different times. These are short research defaults,
not evidence of convergence.

Outputs in the run directory:

- `experiment.json`, `manifest.json`: settings, matched model configs, checkpoint
  metadata, and parameter counts.
- `training.csv`, `checkpoints/*.pt`: per-update training metrics and initial,
  periodic, and final snapshots of both models.
- `matches.csv`, `elo.csv`: all scored checkpoint pairings and fitted ratings.
- `elo_over_time.png`, `elo_over_time.svg`: checkpoint Elo against training
  decisions per learner, with mean and standard deviation across seeds.

After training, saved checkpoints play either the configured balanced graph or
the full round-robin tournament, in both controller-port orders. Two warmup bouts
precede four scored bouts per side;
context persists within each side's series, then resets. Actions are sampled
with recorded reproducible pairing seeds. Warmup games do not affect scores.
Ratings are fitted jointly with a regularized Bradley–Terry model on the Elo
scale, centered at 1500 across the entire checkpoint population. This keeps all
points in one common tournament rather than resetting ratings at each training
update. The shaded region shows variation across training seeds, not a confidence
interval. Ratings are relative to this checkpoint population and may shift if
more checkpoints are added. A full tournament costs quadratically more as more
checkpoints are saved; increase `checkpoint_interval` to keep evaluation smaller.

Train without a tournament, evaluate existing snapshots, or redraw an existing graph:

```bash
bash scripts/run_melee_comparison.sh --run-dir runs/melee_first_hit --train-only
../.venv/bin/python -m latent_policy.melee_elo runs/melee_first_hit
../.venv/bin/python -m latent_policy.melee_elo runs/melee_first_hit --plot-only
```

Use a new run directory for each training experiment. Existing result directories
are rejected to avoid mixing runs. The previous Melee Light source, including
local script/test edits, is backed up locally in `runs/legacy_melee_light_source.zip`.
Historical results in `EXPERIMENTS.md` and `runs/` belong to the removed environment
and cannot be compared to these real-Melee experiments.

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
