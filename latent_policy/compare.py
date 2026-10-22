from __future__ import annotations

import argparse
import copy
import csv
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from latent_policy.models import PolicyConfig, build_policy, count_parameters
from latent_policy.ppo import TrainConfig, load_config, make_env, make_policy_config, train
from latent_policy.utils import write_json


def parameter_count(cfg: PolicyConfig) -> int:
    with torch.device("meta"):
        return count_parameters(build_policy(cfg))


def match_baseline(latent_cfg: PolicyConfig, tolerance: float = 0.01) -> PolicyConfig:
    if latent_cfg.agent not in {"hyper_head", "full_hyper", "film"}:
        raise ValueError("the latent agent must be hyper_head, full_hyper, or film")
    baseline = replace(latent_cfg, agent="static_mlp", encoder="mean")
    return match_parameter_budget(baseline, parameter_count(latent_cfg), tolerance)


def match_parameter_budget(template: PolicyConfig, target: int, tolerance: float = 0.01) -> PolicyConfig:
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("parameter tolerance must be between 0 and 1")
    if target <= 0:
        raise ValueError("parameter target must be positive")

    def count_width(width: int) -> int:
        return parameter_count(replace(template, hidden_dim=width))

    lower, upper = 1, 1
    while count_width(upper) < target:
        upper *= 2
    while lower < upper:
        middle = (lower + upper) // 2
        if count_width(middle) < target:
            lower = middle + 1
        else:
            upper = middle
    candidates = {max(1, lower - 1), lower}
    width = min(candidates, key=lambda candidate: (abs(count_width(candidate) - target), candidate))
    actual = count_width(width)
    gap = abs(actual - target) / target
    if gap > tolerance:
        raise ValueError(
            f"closest {template.agent} has {actual:,} parameters (hidden_dim={width}), "
            f"versus {target:,} target parameters: gap {gap:.2%} exceeds {tolerance:.2%} tolerance"
        )
    return replace(template, hidden_dim=width)


def compare(
    base: TrainConfig,
    seeds: list[int],
    run_dir: str | Path,
    tolerance: float = 0.01,
    dry_run: bool = False,
) -> list[dict[str, object]]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least one seed, without duplicates")
    if base.total_updates <= 0 or base.eval_episodes <= 0:
        raise ValueError("updates and evaluation episodes must be positive")
    probe = make_env(base, eval_envs=1)
    try:
        latent_cfg = make_policy_config(base, probe)
    finally:
        if hasattr(probe, "close"):
            probe.close()
    baseline_cfg = match_baseline(latent_cfg, tolerance)
    latent_params = parameter_count(latent_cfg)
    baseline_params = parameter_count(baseline_cfg)
    gap = abs(baseline_params - latent_params) / latent_params
    print(f"{latent_cfg.agent}: hidden_dim={latent_cfg.hidden_dim}, params={latent_params:,}")
    print(f"static_mlp: hidden_dim={baseline_cfg.hidden_dim}, params={baseline_params:,}; gap={gap:.3%}")

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    effective = copy.deepcopy(base)
    effective.eval_interval = max(1, effective.eval_interval)
    write_json(output / "comparison_plan.json", {
        "training_config": asdict(effective),
        "seeds": seeds,
        "latent_policy": asdict(latent_cfg),
        "baseline_policy": asdict(baseline_cfg),
        "latent_params": latent_params,
        "baseline_params": baseline_params,
        "relative_param_gap": gap,
        "param_tolerance": tolerance,
    })
    if dry_run:
        return []

    rows: list[dict[str, object]] = []
    for seed in seeds:
        for policy_cfg in (baseline_cfg, latent_cfg):
            cfg = copy.deepcopy(effective)
            cfg.seed = seed
            cfg.policy.agent = policy_cfg.agent
            cfg.policy.encoder = policy_cfg.encoder
            cfg.policy.hidden_dim = policy_cfg.hidden_dim
            cfg.run_dir = str(output)
            cfg.run_name = f"{policy_cfg.agent}_{policy_cfg.encoder}_h{policy_cfg.hidden_dim}_seed{seed}"
            metrics = train(cfg)
            expected_params = parameter_count(policy_cfg)
            if metrics["params"] != expected_params:
                raise RuntimeError("trained model parameter count differs from comparison plan")
            row = {
                **metrics,
                "agent": policy_cfg.agent,
                "encoder": policy_cfg.encoder,
                "hidden_dim": policy_cfg.hidden_dim,
                "seed": seed,
                "relative_param_gap": gap,
            }
            rows.append(row)
            with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({key for item in rows for key in item}))
                writer.writeheader()
                writer.writerows(rows)

    paired_deltas = []
    for baseline_row, latent_row in zip(rows[::2], rows[1::2]):
        delta = float(latent_row["eval_return_mean"]) - float(baseline_row["eval_return_mean"])
        paired_deltas.append(delta)
        print(f"seed={baseline_row['seed']}: latent - baseline eval return = {delta:+.4f}")
    for agent in ("static_mlp", latent_cfg.agent):
        returns = [float(row["eval_return_mean"]) for row in rows if row["agent"] == agent]
        print(f"{agent}: eval return mean={np.mean(returns):.4f}, std across seeds={np.std(returns):.4f}")
    print(f"Mean paired return difference: {np.mean(paired_deltas):+.4f}")
    print(f"wrote {output / 'summary.csv'}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare a latent policy with a parameter-matched static MLP.")
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--latent-agent", choices=["hyper_head", "full_hyper", "film"], default=None,
                        help="Override the latent architecture from the config.")
    parser.add_argument("--encoder", choices=["flat", "mean", "gru", "attention"], default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-dir", default="runs/parameter_matched")
    parser.add_argument("--param-tolerance", type=float, default=0.01,
                        help="Maximum absolute parameter gap / latent parameter count (default: 0.01 = 1%%).")
    parser.add_argument("--dry-run", action="store_true", help="Print counts and save the plan without training.")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.latent_agent is not None:
        cfg.policy.agent = args.latent_agent
    if args.encoder is not None:
        cfg.policy.encoder = args.encoder
    if args.updates is not None:
        cfg.total_updates = args.updates
    if args.eval_episodes is not None:
        cfg.eval_episodes = args.eval_episodes
    if args.device is not None:
        cfg.device = args.device
    if args.no_progress:
        cfg.progress = False
    try:
        compare(cfg, args.seeds, args.run_dir, args.param_tolerance, args.dry_run)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
