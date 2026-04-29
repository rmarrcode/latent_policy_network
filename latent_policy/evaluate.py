from __future__ import annotations

import argparse

import torch
import yaml

from latent_policy.evaluation import evaluate_policy_in_env
from latent_policy.ppo import load_checkpoint, make_env
from latent_policy.utils import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a latent-policy checkpoint.")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of argmax.")
    args = parser.parse_args()

    device = select_device(args.device)
    policy, cfg, policy_cfg, _ = load_checkpoint(args.checkpoint, device)
    policy.to(device)
    policy.eval()
    eval_envs = min(max(1, args.episodes), max(1, cfg.public_env.num_envs if cfg.public_env is not None else cfg.env.num_envs))
    env = make_env(cfg, eval_envs=eval_envs, seed_offset=10_000)
    with torch.no_grad():
        metrics = evaluate_policy_in_env(
            policy,
            env,
            context_len=policy_cfg.context_len,
            device=device,
            episodes=args.episodes,
            deterministic=not args.sample,
        )
    print(yaml.safe_dump(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
