from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from latent_policy.melee_env import MeleeConfig, make_melee_env
from latent_policy.melee_self_play import RuntimePolicy, SelfPlayConfig, write_csv
from latent_policy.models import PolicyConfig, build_policy, count_parameters
from latent_policy.utils import select_device, set_seed, write_json


def fit_elo(competitors: int, results: list[tuple[int, int, float, int]], regularization: float = 1.0) -> np.ndarray:
    if competitors < 2 or regularization <= 0:
        raise ValueError("Elo fitting requires at least two competitors and positive regularization")
    strength = np.zeros(competitors, dtype=np.float64)
    for _ in range(100):
        gradient = regularization * strength
        hessian = np.eye(competitors) * regularization
        for first, second, score, games in results:
            probability = 1 / (1 + np.exp(np.clip(strength[second] - strength[first], -50, 50)))
            error = games * probability - score
            curvature = games * probability * (1 - probability)
            gradient[first] += error
            gradient[second] -= error
            hessian[first, first] += curvature
            hessian[second, second] += curvature
            hessian[first, second] -= curvature
            hessian[second, first] -= curvature
        step = np.linalg.solve(hessian, gradient)
        strength -= step / max(1.0, float(np.max(np.abs(step))))
        if np.max(np.abs(gradient)) < 1e-8:
            break
    return 1500 + (strength - strength.mean()) * 400 / np.log(10)


def play_bout(env, runtimes):
    obs = env.reset()
    while True:
        actions = [runtime.act(obs[side]) for side, runtime in enumerate(runtimes)]
        obs, rewards, done, info = env.step(actions)
        for side, runtime in enumerate(runtimes):
            runtime.observe(obs[side])
        if done:
            return rewards, info


def play_series(env, first, second, options: SelfPlayConfig) -> tuple[float, int]:
    runtimes = [RuntimePolicy(first), RuntimePolicy(second)]
    score = 0.0
    frames = 0
    for bout in range(options.eval_warmup_bouts + options.eval_games_per_side):
        rewards, info = play_bout(env, runtimes)
        if bout >= options.eval_warmup_bouts:
            score += (float(rewards[0]) + 1) / 2
            frames += info["frames"]
    return score, frames


def select_pairings(competitors: int, opponents_per_checkpoint: int) -> list[tuple[int, int]]:
    if opponents_per_checkpoint < 0 or opponents_per_checkpoint % 2:
        raise ValueError("opponents_per_checkpoint must be zero or a positive even number")
    rng = np.random.default_rng(10_000)
    if opponents_per_checkpoint == 0 or opponents_per_checkpoint >= competitors - 1:
        pairs = list(itertools.combinations(range(competitors), 2))
    else:
        order = rng.permutation(competitors).tolist()
        edges = set()
        for distance in range(1, opponents_per_checkpoint // 2 + 1):
            for index, first in enumerate(order):
                second = order[(index + distance) % competitors]
                edges.add(tuple(sorted((first, second))))
        pairs = sorted(edges)
    rng.shuffle(pairs)
    return pairs


def evaluate_switches(env, policies, manifest, options: SelfPlayConfig, output: Path) -> None:
    seeds = sorted({entry["seed"] for entry in manifest})
    final_update = max(entry["update"] for entry in manifest)
    final_indices = [index for index, entry in enumerate(manifest) if entry["update"] == final_update]
    rows = []
    write_json(output / "switch_protocol.json", {
        "held_out_training_seed": len(seeds) > 1, "warmup_bouts": options.eval_warmup_bouts,
        "scored_bouts_after_switch": options.eval_switch_games, "reset_learner_context_on_switch": False,
        "balanced_controller_ports": True,
    })
    for learner_index in final_indices:
        learner_entry = manifest[learner_index]
        seed_index = seeds.index(learner_entry["seed"])
        target_seed = seeds[(seed_index + 1) % len(seeds)]
        targets = [index for index in final_indices if manifest[index]["seed"] == target_seed]
        old_index = next(index for index, entry in enumerate(manifest)
                         if entry["seed"] == target_seed and entry["update"] == 0)
        for target_index in targets:
            for side in (0, 1):
                set_seed(20_000 + seed_index * 100 + targets.index(target_index) * 2 + side)
                learner = RuntimePolicy(policies[learner_index])
                old_opponent = RuntimePolicy(policies[old_index])
                runtimes = [learner, old_opponent] if side == 0 else [old_opponent, learner]
                for _ in range(options.eval_warmup_bouts):
                    play_bout(env, runtimes)
                new_opponent = RuntimePolicy(policies[target_index])
                runtimes[1 - side] = new_opponent
                for bout in range(options.eval_switch_games):
                    rewards, info = play_bout(env, runtimes)
                    rows.append({"agent": learner_entry["agent"], "seed": learner_entry["seed"],
                                 "checkpoint": learner_entry["id"], "old_opponent": manifest[old_index]["id"],
                                 "new_opponent": manifest[target_index]["id"], "target_seed": target_seed,
                                 "port": side + 1, "bout_since_switch": bout,
                                 "score": (float(rewards[side]) + 1) / 2, "frames": info["frames"],
                                 "reason": info.get("reason", "unspecified")})
                write_csv(output / "switch_bouts.csv", rows)
        print(f"Switch evaluation complete: {learner_entry['id']}", flush=True)
    summary = []
    for agent in sorted({row["agent"] for row in rows}):
        for bout in range(options.eval_switch_games):
            scores = [row["score"] for row in rows if row["agent"] == agent and row["bout_since_switch"] == bout]
            summary.append({"agent": agent, "bout_since_switch": bout, "games": len(scores),
                            "mean_score": float(np.mean(scores)), "score_std": float(np.std(scores))})
    write_csv(output / "switch_summary.csv", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    for agent in sorted({row["agent"] for row in summary}):
        points = [row for row in summary if row["agent"] == agent]
        axis.plot([row["bout_since_switch"] for row in points], [row["mean_score"] for row in points],
                  marker="o", label=agent)
    axis.set(xlabel="Bout since hidden opponent switch (0 = first)", ylabel="Mean score (draw = 0.5)",
             ylim=(0, 1), title="Final checkpoints: " + ("held-out training-seed opponents" if len(seeds) > 1 else "same-seed opponents"))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "switch_adaptation.png", dpi=180)
    plt.close(figure)


def plot_elo(run_dir: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(run_dir)
    with (output / "elo.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No Elo results to plot")
    figure, axis = plt.subplots(figsize=(9, 5))
    for agent in sorted({row["agent"] for row in rows}):
        agent_rows = [row for row in rows if row["agent"] == agent]
        steps = sorted({int(row["env_steps"]) for row in agent_rows})
        means, deviations = [], []
        for step in steps:
            values = [float(row["elo"]) for row in agent_rows if int(row["env_steps"]) == step]
            means.append(np.mean(values))
            deviations.append(np.std(values))
        means, deviations = np.asarray(means), np.asarray(deviations)
        line, = axis.plot(steps, means, marker="o", label=agent)
        if len({row["seed"] for row in agent_rows}) > 1:
            axis.fill_between(steps, means - deviations, means + deviations,
                              color=line.get_color(), alpha=0.15)
    axis.axhline(1500, color="grey", linewidth=0.8, linestyle="--")
    axis.set(xlabel="Training decisions per learner", ylabel="Elo (pooled checkpoint tournament)",
             title="Real Melee: first-hit self-play, matched parameter counts")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    path = output / "elo_over_time.png"
    figure.savefig(path, dpi=180)
    figure.savefig(output / "elo_over_time.svg")
    plt.close(figure)
    return path


def evaluate_checkpoints(run_dir: str | Path, device: str = "auto", env_factory=make_melee_env,
                         iso_path: str | None = None, dolphin_path: str | None = None,
                         backend: str | None = None, data_dir: str | None = None) -> list[dict]:
    output = Path(run_dir)
    experiment = json.loads((output / "experiment.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    if len(manifest) < 2 or len({entry["id"] for entry in manifest}) != len(manifest):
        raise ValueError("A tournament needs at least two uniquely identified checkpoints")
    melee_cfg = MeleeConfig(**experiment["melee"])
    if iso_path is not None:
        melee_cfg.iso_path = iso_path
    if dolphin_path is not None:
        melee_cfg.dolphin_path = dolphin_path
    if backend is not None:
        melee_cfg.backend = backend
    if data_dir is not None:
        melee_cfg.data_dir = data_dir
    options = SelfPlayConfig(**experiment["self_play"])
    torch.set_num_threads(max(1, experiment["training"]["torch_threads"]))
    policies = []
    for entry in manifest:
        checkpoint = torch.load(entry["path"], map_location=select_device(device), weights_only=False)
        policy = build_policy(PolicyConfig(**checkpoint["policy_config"])).to(select_device(device))
        policy.load_state_dict(checkpoint["model_state"])
        if count_parameters(policy) != entry["params"]:
            raise ValueError("Checkpoint count differs from manifest")
        policies.append(policy.eval())
    matches, results = [], []
    pairs = select_pairings(len(manifest), options.eval_opponents_per_checkpoint)
    coverage = [sum(index in pair for pair in pairs) for index in range(len(manifest))]
    write_json(output / "evaluation_plan.json", {
        "strategy": "full_round_robin" if len(pairs) == len(manifest) * (len(manifest) - 1) // 2 else "balanced_connected_graph",
        "pairings": len(pairs), "seed": 10_000,
        "opponents_per_checkpoint": {entry["id"]: coverage[index] for index, entry in enumerate(manifest)},
        "scored_games_per_pairing": options.eval_games_per_side * 2,
        "warmup_bouts_per_side": options.eval_warmup_bouts,
    })
    env = env_factory(melee_cfg)
    try:
        for pair_index, (first, second) in enumerate(pairs):
            set_seed(10_000 + pair_index)
            forward_score, forward_frames = play_series(env, policies[first], policies[second], options)
            set_seed(10_000 + pair_index)
            reverse_score, reverse_frames = play_series(env, policies[second], policies[first], options)
            score = forward_score + options.eval_games_per_side - reverse_score
            games = options.eval_games_per_side * 2
            results.append((first, second, score, games))
            matches.append({"first": manifest[first]["id"], "second": manifest[second]["id"],
                            "pair_seed": 10_000 + pair_index,
                            "first_score": score, "games": games, "first_score_as_p1": forward_score,
                            "first_score_as_p2": options.eval_games_per_side - reverse_score,
                            "scored_frames": forward_frames + reverse_frames})
            write_csv(output / "matches.csv", matches)
            print(f"Elo pairing {pair_index + 1}/{len(pairs)}: "
                  f"{manifest[first]['id']} vs {manifest[second]['id']}: {score:g}/{games}", flush=True)
        ratings = fit_elo(len(manifest), results)
        rows = [{**entry, "elo": float(ratings[index])} for index, entry in enumerate(manifest)]
        write_csv(output / "elo.csv", rows)
        print(f"wrote {plot_elo(output)}", flush=True)
        if options.eval_switch_games:
            evaluate_switches(env, policies, manifest, options, output)
    finally:
        env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rate real-Melee checkpoints and graph Elo over training steps.")
    parser.add_argument("run_dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iso", default=None)
    parser.add_argument("--dolphin", default=None)
    parser.add_argument("--backend", choices=["decomp", "slippi"], default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        print(plot_elo(args.run_dir))
    else:
        evaluate_checkpoints(args.run_dir, device=args.device, iso_path=args.iso, dolphin_path=args.dolphin,
                             backend=args.backend, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
