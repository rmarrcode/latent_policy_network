from __future__ import annotations

import argparse
import csv
import copy
from pathlib import Path

from latent_policy.ppo import load_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small architecture sweep.")
    parser.add_argument("--config", type=str, default="configs/quick.yaml")
    parser.add_argument("--agents", nargs="+", default=["static_mlp", "hyper_head", "film", "full_hyper"])
    parser.add_argument("--encoders", nargs="+", default=["gru"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--run-dir", type=str, default="runs/sweep")
    parser.add_argument("--summary", type=str, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    summary_path = Path(args.summary or Path(args.run_dir) / "summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for seed in args.seeds:
        for agent in args.agents:
            encoders = ["mean"] if agent == "static_mlp" else args.encoders
            for encoder in encoders:
                cfg = copy.deepcopy(base)
                cfg.seed = seed
                cfg.policy.agent = agent
                cfg.policy.encoder = encoder
                cfg.run_dir = args.run_dir
                cfg.run_name = f"{agent}_{encoder}_seed{seed}"
                cfg.progress = not args.no_progress and base.progress
                if args.updates is not None:
                    cfg.total_updates = args.updates
                metrics = train(cfg)
                row = {
                    "agent": agent,
                    "encoder": encoder,
                    "seed": seed,
                    **metrics,
                }
                rows.append(row)
                fieldnames = list(rows[0].keys())
                with summary_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)

    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
