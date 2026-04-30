from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass, field
from pathlib import Path

from latent_policy.ppo import TrainConfig, train
from latent_policy.public_envs import PublicEnvConfig


@dataclass(frozen=True)
class PublicSpec:
    env_id: str
    kind: str
    name: str
    train_agent: str | None = None
    num_envs: int = 16
    episode_length: int = 64
    max_cycles: int = 64
    hidden_dim: int = 48
    latent_dim: int = 48
    generated_hidden_dim: int = 16
    opponent_pool: tuple[str, ...] | None = None
    env_kwargs: dict[str, object] = field(default_factory=dict)


PUBLIC_SPECS: dict[str, PublicSpec] = {
    "openspiel_matrix_rps": PublicSpec("openspiel_matrix_rps", "openspiel_matrix", "matrix_rps", num_envs=32),
    "openspiel_matrix_rpsw": PublicSpec("openspiel_matrix_rpsw", "openspiel_matrix", "matrix_rpsw", num_envs=32),
    "openspiel_matrix_pd": PublicSpec("openspiel_matrix_pd", "openspiel_matrix", "matrix_pd", num_envs=32),
    "openspiel_kuhn": PublicSpec("openspiel_kuhn", "openspiel_turn", "kuhn_poker", num_envs=32, episode_length=24),
    "openspiel_tictactoe": PublicSpec("openspiel_tictactoe", "openspiel_turn", "tic_tac_toe", num_envs=24, episode_length=18),
    "openspiel_connect_four": PublicSpec("openspiel_connect_four", "openspiel_turn", "connect_four", num_envs=16, episode_length=42),
    "pettingzoo_rps": PublicSpec("pettingzoo_rps", "pettingzoo_parallel", "pettingzoo.classic.rps_v2:parallel_env", train_agent="player_0", num_envs=32),
    "mpe_simple_adversary": PublicSpec("mpe_simple_adversary", "pettingzoo_parallel", "mpe2.simple_adversary_v3:parallel_env", train_agent="adversary_0", num_envs=12),
    "mpe_simple_push": PublicSpec("mpe_simple_push", "pettingzoo_parallel", "mpe2.simple_push_v3:parallel_env", train_agent="adversary_0", num_envs=12),
    "mpe_simple_tag": PublicSpec("mpe_simple_tag", "pettingzoo_parallel", "mpe2.simple_tag_v3:parallel_env", train_agent="adversary_0", num_envs=8),
    "mpe_simple_world_comm": PublicSpec("mpe_simple_world_comm", "pettingzoo_parallel", "mpe2.simple_world_comm_v3:parallel_env", train_agent="adversary_0", num_envs=6, episode_length=96, max_cycles=96),
    "slime_volley": PublicSpec("slime_volley", "gym_single", "SlimeVolley-v0", num_envs=8, episode_length=300, max_cycles=300),
    "melee_light_knockback": PublicSpec(
        "melee_light_knockback",
        "gym_single",
        "melee_light_knockback",
        num_envs=1,
        episode_length=60,
        max_cycles=60,
        hidden_dim=32,
        latent_dim=32,
        env_kwargs={"frame_skip": 4, "max_episode_frames": 240, "opponent_level": 3},
    ),
    "footsies": PublicSpec("footsies", "gym_single", "footsies", num_envs=2, episode_length=300, max_cycles=300, hidden_dim=32, latent_dim=32),
    "magent_battle": PublicSpec("magent_battle", "pettingzoo_parallel", "magent2.environments.battle_v4:parallel_env", train_agent="red_0", num_envs=2, episode_length=32, max_cycles=32, hidden_dim=32, latent_dim=32, generated_hidden_dim=8),
}


def make_config(
    spec: PublicSpec,
    agent: str,
    encoder: str,
    seed: int,
    updates: int,
    run_dir: str,
    no_progress: bool,
    opponent_pool: tuple[str, ...] | None = None,
    switch_hazard: float = 0.05,
    min_switch_interval: int = 8,
) -> TrainConfig:
    cfg = TrainConfig()
    cfg.seed = seed
    cfg.run_dir = run_dir
    cfg.run_name = f"{spec.env_id}_{agent}_{encoder}_seed{seed}"
    cfg.total_updates = updates
    cfg.num_steps = min(64, spec.episode_length)
    cfg.num_minibatches = 4
    cfg.update_epochs = 3
    cfg.eval_interval = max(1, updates)
    cfg.eval_episodes = min(64, spec.num_envs * 4)
    cfg.save_interval = updates
    cfg.progress = not no_progress
    cfg.torch_threads = 4
    cfg.policy.agent = agent
    cfg.policy.encoder = encoder
    cfg.policy.context_len = 12
    cfg.policy.hidden_dim = spec.hidden_dim
    cfg.policy.latent_dim = spec.latent_dim
    cfg.policy.generated_hidden_dim = spec.generated_hidden_dim
    if agent == "static_mlp":
        cfg.policy.encoder = "mean"
    cfg.public_env = PublicEnvConfig(
        kind=spec.kind,
        name=spec.name,
        num_envs=spec.num_envs,
        episode_length=spec.episode_length,
        max_cycles=spec.max_cycles,
        train_agent=spec.train_agent,
        seed=seed,
        switch_hazard=switch_hazard,
        min_switch_interval=min_switch_interval,
        opponent_pool=tuple(opponent_pool or spec.opponent_pool or PublicEnvConfig().opponent_pool),
        env_kwargs=copy.deepcopy(spec.env_kwargs),
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quick experiments across public env adapters.")
    parser.add_argument("--envs", nargs="+", default=list(PUBLIC_SPECS))
    parser.add_argument("--agents", nargs="+", default=["static_mlp", "hyper_head", "full_hyper"])
    parser.add_argument("--encoder", default="gru")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--run-dir", type=str, default="runs/public_suite")
    parser.add_argument("--summary", type=str, default=None)
    parser.add_argument("--opponent-pool", nargs="+", default=None)
    parser.add_argument("--switch-hazard", type=float, default=0.05)
    parser.add_argument("--min-switch-interval", type=int, default=8)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    summary_path = Path(args.summary or Path(args.run_dir) / "summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    if summary_path.exists() and summary_path.stat().st_size > 0:
        with summary_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    def row_key(row: dict[str, object]) -> tuple[object, ...]:
        return (
            row.get("env_id"),
            row.get("agent"),
            row.get("encoder"),
            row.get("seed"),
        )

    row_index = {row_key(row): i for i, row in enumerate(rows)}
    for env_id in args.envs:
        spec = PUBLIC_SPECS[env_id]
        for seed in args.seeds:
            for agent in args.agents:
                encoder = "mean" if agent == "static_mlp" else args.encoder
                cfg = make_config(
                    copy.deepcopy(spec),
                    agent,
                    encoder,
                    seed,
                    args.updates,
                    args.run_dir,
                    args.no_progress,
                    tuple(args.opponent_pool) if args.opponent_pool else None,
                    args.switch_hazard,
                    args.min_switch_interval,
                )
                try:
                    metrics = train(cfg)
                    row = {
                        "status": "ok",
                        "env_id": env_id,
                        "kind": spec.kind,
                        "name": spec.name,
                        "agent": agent,
                        "encoder": encoder,
                        "seed": seed,
                        **metrics,
                    }
                except Exception as exc:
                    row = {
                        "status": "error",
                        "env_id": env_id,
                        "kind": spec.kind,
                        "name": spec.name,
                        "agent": agent,
                        "encoder": encoder,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                key = row_key(row)
                if key in row_index:
                    rows[row_index[key]] = row
                else:
                    row_index[key] = len(rows)
                    rows.append(row)
                fieldnames = sorted({key for item in rows for key in item})
                with summary_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"{row['status']}: {env_id} {agent} seed={seed}")

    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
