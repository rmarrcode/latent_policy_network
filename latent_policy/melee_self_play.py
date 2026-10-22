from __future__ import annotations

import argparse
import copy
import csv
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from latent_policy.compare import match_baseline, match_parameter_budget, parameter_count
from latent_policy.evaluation import append_context
from latent_policy.melee_env import ACTION_NAMES, OBS_DIM, MeleeConfig, make_melee_env
from latent_policy.models import PolicyConfig, build_policy, count_parameters
from latent_policy.ppo import TrainConfig, config_from_dict
from latent_policy.utils import CSVLogger, select_device, set_seed, write_json


@dataclass
class SelfPlayConfig:
    agents: tuple[str, ...] | None = None
    seeds: tuple[int, ...] = (1, 2, 3)
    param_tolerance: float = 0.01
    checkpoint_interval: int = 10
    pool_size_per_agent: int = 4
    opponent_bouts: int = 4
    eval_games_per_side: int = 4
    eval_warmup_bouts: int = 2
    eval_opponents_per_checkpoint: int = 0
    eval_switch_games: int = 0


class RuntimePolicy:
    def __init__(self, policy):
        self.policy = policy
        for module in policy.modules():
            if isinstance(module, torch.nn.RNNBase):
                module.flatten_parameters()
        self.device = next(policy.parameters()).device
        self.context = torch.zeros(1, policy.cfg.context_len, policy.cfg.obs_dim, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        return int(self.policy.act(torch.as_tensor(obs, device=self.device).unsqueeze(0),
                                   self.context, deterministic=False).item())

    def observe(self, obs: np.ndarray) -> None:
        self.context = append_context(self.context, torch.as_tensor(obs, device=self.device).unsqueeze(0))


def policy_configs(cfg: TrainConfig, options: SelfPlayConfig) -> list[PolicyConfig]:
    latent = PolicyConfig(obs_dim=OBS_DIM, action_dim=len(ACTION_NAMES), **asdict(cfg.policy))
    baseline = match_baseline(latent, options.param_tolerance)
    if options.agents is None:
        return [baseline, latent]
    if len(set(options.agents)) != len(options.agents) or not options.agents:
        raise ValueError("comparison agents must be nonempty and unique")
    if cfg.policy.agent not in options.agents:
        raise ValueError("comparison agents must include the target latent architecture")
    latent = replace(latent, encoder_hidden_dim=latent.encoder_hidden_dim or latent.hidden_dim)
    target = parameter_count(latent)
    configs = []
    for agent in options.agents:
        template = replace(latent, agent=agent, encoder="mean" if agent == "static_mlp" else latent.encoder)
        configs.append(match_parameter_budget(template, target, options.param_tolerance))
    return configs


def validate(cfg: TrainConfig, options: SelfPlayConfig) -> None:
    if cfg.target_kl is not None:
        raise ValueError("Set target_kl: null so both learners receive the full PPO optimization budget")
    if cfg.total_updates <= 0 or cfg.num_steps < 2:
        raise ValueError("total_updates must be positive and num_steps must be at least two")
    if not 1 <= cfg.num_minibatches <= cfg.num_steps // 2 or cfg.update_epochs <= 0:
        raise ValueError("PPO needs positive epochs and at least two samples per minibatch")
    if not options.seeds or len(set(options.seeds)) != len(options.seeds):
        raise ValueError("seeds must be nonempty and unique")
    if min(options.checkpoint_interval, options.pool_size_per_agent, options.opponent_bouts,
           options.eval_games_per_side) <= 0 or options.eval_warmup_bouts < 0:
        raise ValueError("self-play intervals, pool size, and games must be positive; warmup must be nonnegative")
    if options.eval_opponents_per_checkpoint < 0 or options.eval_opponents_per_checkpoint % 2:
        raise ValueError("eval_opponents_per_checkpoint must be zero or a positive even number")
    if options.eval_switch_games < 0:
        raise ValueError("eval_switch_games must be nonnegative")


@torch.no_grad()
def benchmark_policy(policy, iterations: int = 30) -> dict[str, float]:
    runtime = RuntimePolicy(policy)
    obs = np.zeros(policy.cfg.obs_dim, dtype=np.float32)
    for _ in range(5):
        runtime.act(obs)
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    start = time.perf_counter()
    for _ in range(iterations):
        runtime.act(obs)
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    return {"batch_one_action_ms": (time.perf_counter() - start) * 1000 / iterations}


def frozen_snapshot(policy, update: int):
    snapshot = copy.deepcopy(policy).eval()
    snapshot.self_play_update = update
    return snapshot


def collect_rollout(env, policy, opponents, cfg: TrainConfig, options: SelfPlayConfig, seed: int):
    device = next(policy.parameters()).device
    rng = np.random.default_rng(seed)
    learner = RuntimePolicy(policy)
    opponent = None
    side = int(rng.integers(2))
    opponent_index = None
    bouts = options.opponent_bouts
    obs = env.reset()
    records: dict[str, list] = {key: [] for key in ("obs", "context", "actions", "logprobs", "values", "rewards", "done")}
    outcomes = []
    bout_records = []
    age_rewards = {"0_3": [], "4_15": [], "16_plus": []}
    bout_returns = {"0": [], "1": [], "2_plus": []}
    opponent_steps = 0
    frames = 0
    for rollout_step in range(cfg.num_steps):
        if bouts >= options.opponent_bouts:
            candidates = [index for index in range(len(opponents)) if index != opponent_index]
            opponent_index = int(rng.choice(candidates or [opponent_index]))
            opponent = RuntimePolicy(opponents[opponent_index])
            bouts = 0
            opponent_steps = 0
        obs_tensor = torch.as_tensor(obs[side], device=device).unsqueeze(0)
        with torch.no_grad():
            action, logprob, _, value = policy.get_action_and_value(obs_tensor, learner.context)
        records["obs"].append(obs_tensor[0])
        records["context"].append(learner.context[0].clone())
        records["actions"].append(action[0])
        records["logprobs"].append(logprob[0])
        records["values"].append(value[0])
        actions = [0, 0]
        actions[side] = int(action.item())
        actions[1 - side] = opponent.act(obs[1 - side])
        old_frames = env.elapsed
        obs, rewards, done, info = env.step(actions)
        frames += env.elapsed - old_frames
        records["rewards"].append(torch.tensor(float(rewards[side]), device=device))
        records["done"].append(torch.tensor(float(done), device=device))
        age_group = "0_3" if opponent_steps < 4 else "4_15" if opponent_steps < 16 else "16_plus"
        age_rewards[age_group].append(float(rewards[side]))
        opponent_steps += 1
        learner.observe(obs[side])
        opponent.observe(obs[1 - side])
        if done:
            outcomes.append(float(rewards[side]))
            bout_group = str(bouts) if bouts < 2 else "2_plus"
            bout_returns[bout_group].append(float(rewards[side]))
            bout_records.append({"opponent_agent": opponent.policy.cfg.agent,
                                 "opponent_update": getattr(opponent.policy, "self_play_update", 0),
                                 "bout_since_switch": bouts, "port": side + 1,
                                 "return": float(rewards[side]), "frames": info["frames"],
                                 "reason": info.get("reason", "unspecified")})
            bouts += 1
            if rollout_step + 1 < cfg.num_steps:
                obs = env.reset()
    with torch.no_grad():
        _, bootstrap = policy.logits_and_value(torch.as_tensor(obs[side], device=device).unsqueeze(0), learner.context)
    batch = {key: torch.stack(values) for key, values in records.items()}
    advantages = torch.zeros_like(batch["rewards"])
    last_advantage = torch.zeros((), device=device)
    for step in reversed(range(cfg.num_steps)):
        next_value = bootstrap.squeeze() if step == cfg.num_steps - 1 else batch["values"][step + 1]
        continuation = 1.0 - batch["done"][step]
        delta = batch["rewards"][step] + cfg.gamma * next_value * continuation - batch["values"][step]
        last_advantage = delta + cfg.gamma * cfg.gae_lambda * continuation * last_advantage
        advantages[step] = last_advantage
    batch["advantages"] = advantages
    batch["returns"] = advantages + batch["values"]
    metrics = {
        "rollout_bouts": len(outcomes), "rollout_return_mean": float(np.mean(outcomes)) if outcomes else float("nan"),
        "rollout_frames": frames, "bout_records": bout_records,
    }
    for prefix, groups in (("reward_age", age_rewards), ("return_bout", bout_returns)):
        for name, values in groups.items():
            metrics[f"rollout_{prefix}_{name}"] = float(np.mean(values)) if values else float("nan")
            metrics[f"rollout_{prefix}_{name}_count"] = len(values)
    return batch, metrics


def ppo_update(policy, optimizer, batch, cfg: TrainConfig) -> dict[str, float]:
    final_metrics = {}
    for _ in range(cfg.update_epochs):
        permutation = torch.randperm(cfg.num_steps, device=batch["obs"].device)
        for indices in torch.tensor_split(permutation, cfg.num_minibatches):
            _, logprob, entropy, value = policy.get_action_and_value(
                batch["obs"][indices], batch["context"][indices], batch["actions"][indices])
            ratio = (logprob - batch["logprobs"][indices]).exp()
            advantage = batch["advantages"][indices]
            if cfg.norm_adv:
                advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
            policy_loss = torch.maximum(-advantage * ratio,
                                        -advantage * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
            value_error = (value - batch["returns"][indices]).square()
            if cfg.clip_vloss:
                clipped = batch["values"][indices] + (value - batch["values"][indices]).clamp(-cfg.clip_coef, cfg.clip_coef)
                value_error = torch.maximum(value_error, (clipped - batch["returns"][indices]).square())
            value_loss = value_error.mean() * 0.5
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite PPO gradients")
            optimizer.step()
            final_metrics = {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                             "entropy": entropy.mean().item()}
    return final_metrics


def save_snapshot(output: Path, policy, optimizer, seed: int, update: int, cfg: TrainConfig) -> dict[str, Any]:
    identity = f"{policy.cfg.agent}_seed{seed}_u{update:04d}"
    path = output / "checkpoints" / f"{identity}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    run_config = copy.deepcopy(cfg)
    run_config.seed = seed
    for name, value in asdict(policy.cfg).items():
        if hasattr(run_config.policy, name):
            setattr(run_config.policy, name, value)
    torch.save({"model_state": policy.state_dict(), "optimizer_state": optimizer.state_dict(),
                "policy_config": asdict(policy.cfg), "config": asdict(run_config), "seed": seed,
                "update": update, "env_steps": update * cfg.num_steps}, path)
    return {"id": identity, "path": str(path.resolve()), "agent": policy.cfg.agent, "seed": seed,
            "update": update, "env_steps": update * cfg.num_steps, "params": count_parameters(policy)}


def train_comparison(cfg: TrainConfig, melee_cfg: MeleeConfig, options: SelfPlayConfig,
                     output: str | Path, env_factory=make_melee_env) -> list[dict[str, Any]]:
    validate(cfg, options)
    configs = policy_configs(cfg, options)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or (output / "checkpoints").exists():
        raise ValueError("Output already contains an experiment; choose a new run directory")
    device = select_device(cfg.device)
    torch.set_num_threads(max(1, cfg.torch_threads))
    logger = CSVLogger(output / "training.csv")
    bout_logger = CSVLogger(output / "training_bouts.csv")
    cost_logger = CSVLogger(output / "inference_cost.csv")
    manifest = []
    env = env_factory(melee_cfg)
    try:
        write_json(output / "experiment.json", {"training": asdict(cfg), "melee": asdict(melee_cfg),
                   "self_play": asdict(options), "policies": [asdict(policy_cfg) for policy_cfg in configs],
                   "protocol": {"version": 2, "context_reset": "rollout_boundary_only",
                                "opponent_switch_signaled": False, "pool_frozen_per_update": True,
                                "history_matched_primary_baseline": "context_mlp",
                                "static_mlp_role": "memory_ablation"}})
        for seed in options.seeds:
            if hasattr(env, "seed"):
                env.seed(seed)
            learners, optimizers, pools = [], [], []
            elapsed_by_agent = {policy_cfg.agent: 0.0 for policy_cfg in configs}
            for policy_cfg in configs:
                set_seed(seed)
                policy = build_policy(policy_cfg).to(device)
                optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5)
                learners.append(policy)
                optimizers.append(optimizer)
                pools.append([frozen_snapshot(policy, 0)])
                manifest.append(save_snapshot(output, policy, optimizer, seed, 0, cfg))
                cost_logger.write({"seed": seed, "agent": policy_cfg.agent, "params": count_parameters(policy),
                                   "hidden_dim": policy_cfg.hidden_dim, "encoder_hidden_dim": policy_cfg.encoder_hidden_dim,
                                   "context_len": policy_cfg.context_len, "device": str(device),
                                   **benchmark_policy(policy)})
            write_json(output / "manifest.json", manifest)
            for update in range(1, cfg.total_updates + 1):
                opponents = [snapshot for pool in pools for snapshot in pool]
                offset = (update - 1) % len(learners)
                order = list(range(offset, len(learners))) + list(range(offset))
                for index in order:
                    policy, optimizer = learners[index], optimizers[index]
                    set_seed(seed + update * 10_000)
                    if cfg.anneal_lr:
                        optimizer.param_groups[0]["lr"] = cfg.learning_rate * (1 - (update - 1) / cfg.total_updates)
                    started = time.perf_counter()
                    batch, rollout_metrics = collect_rollout(env, policy, opponents, cfg, options, seed + update * 10_000)
                    rollout_seconds = time.perf_counter() - started
                    optimization_start = time.perf_counter()
                    loss_metrics = ppo_update(policy, optimizer, batch, cfg)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    optimization_seconds = time.perf_counter() - optimization_start
                    elapsed_by_agent[policy.cfg.agent] += time.perf_counter() - started
                    for record in rollout_metrics.pop("bout_records"):
                        bout_logger.write({"seed": seed, "agent": policy.cfg.agent, "update": update, **record})
                    logger.write({"seed": seed, "agent": policy.cfg.agent, "update": update,
                                  "env_steps": update * cfg.num_steps, "params": count_parameters(policy),
                                  "rollout_wall_seconds": rollout_seconds, "optimization_wall_seconds": optimization_seconds,
                                  "cumulative_train_wall_seconds": elapsed_by_agent[policy.cfg.agent],
                                  **rollout_metrics, **loss_metrics})
                    print(f"seed={seed} update={update}/{cfg.total_updates} {policy.cfg.agent}: "
                          f"bouts={rollout_metrics['rollout_bouts']} return={rollout_metrics['rollout_return_mean']:.3f}", flush=True)
                for index, policy in enumerate(learners):
                    pools[index].append(frozen_snapshot(policy, update))
                    pools[index] = pools[index][-options.pool_size_per_agent:]
                    if update % options.checkpoint_interval == 0 or update == cfg.total_updates:
                        manifest.append(save_snapshot(output, policy, optimizers[index], seed, update, cfg))
                write_json(output / "manifest.json", manifest)
    finally:
        env.close()
    return manifest


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_experiment(path: str | Path) -> tuple[TrainConfig, MeleeConfig, SelfPlayConfig]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    data = dict(data)
    melee_cfg = MeleeConfig(**data.pop("melee", {}))
    options = SelfPlayConfig(**data.pop("self_play", {}))
    cfg = config_from_dict(data)
    if cfg.public_env is not None:
        raise ValueError("Self-play uses the melee configuration, not public_env")
    return cfg, melee_cfg, options


def main() -> None:
    from latent_policy.melee_elo import evaluate_checkpoints

    parser = argparse.ArgumentParser(description="Parameter-matched first-hit Melee self-play and checkpoint Elo graph.")
    parser.add_argument("--config", default="configs/melee_self_play.yaml")
    parser.add_argument("--run-dir", default="runs/melee_self_play")
    parser.add_argument("--iso", default=None)
    parser.add_argument("--dolphin", default=None)
    parser.add_argument("--backend", choices=["decomp", "slippi"], default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.train_only and args.eval_only:
        parser.error("--train-only and --eval-only are mutually exclusive")
    try:
        cfg, melee_cfg, options = load_experiment(args.config)
        if args.iso is not None:
            melee_cfg.iso_path = args.iso
        if args.dolphin is not None:
            melee_cfg.dolphin_path = args.dolphin
        if args.backend is not None:
            melee_cfg.backend = args.backend
        if args.data_dir is not None:
            melee_cfg.data_dir = args.data_dir
        if args.device is not None:
            cfg.device = args.device
        if args.updates is not None:
            cfg.total_updates = args.updates
        if args.seeds is not None:
            options.seeds = tuple(args.seeds)
        validate(cfg, options)
        configs = policy_configs(cfg, options)
        counts = [parameter_count(policy_cfg) for policy_cfg in configs]
        target = parameter_count(PolicyConfig(obs_dim=OBS_DIM, action_dim=len(ACTION_NAMES), **asdict(cfg.policy)))
        for policy_cfg, count in zip(configs, counts):
            print(f"{policy_cfg.agent}: params={count:,}, gap={abs(count - target) / target:.3%}, "
                  f"hidden_dim={policy_cfg.hidden_dim}, encoder_hidden_dim={policy_cfg.encoder_hidden_dim}", flush=True)
        if args.dry_run:
            print(yaml.safe_dump({"policies": [asdict(policy_cfg) for policy_cfg in configs],
                                  "self_play": asdict(options)}, sort_keys=False))
            return
        if not args.eval_only:
            train_comparison(cfg, melee_cfg, options, args.run_dir)
        if not args.train_only:
            evaluate_checkpoints(args.run_dir, device=cfg.device, iso_path=args.iso, dolphin_path=args.dolphin,
                                 backend=args.backend, data_dir=args.data_dir)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
