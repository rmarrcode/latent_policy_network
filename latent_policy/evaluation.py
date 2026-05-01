from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from latent_policy.envs import SwitchingDuelVecEnv, SwitchingGameConfig
from latent_policy.models import ActorCritic


def append_context(
    context: torch.Tensor,
    obs: torch.Tensor,
    done: torch.Tensor | None = None,
) -> torch.Tensor:
    context = torch.roll(context, shifts=-1, dims=1)
    context[:, -1, :] = obs
    if done is not None and bool(done.any()):
        context[done.bool()] = 0.0
    return context


@torch.no_grad()
def evaluate_policy(
    policy: ActorCritic,
    env_cfg: SwitchingGameConfig,
    context_len: int,
    device: torch.device,
    episodes: int = 64,
    deterministic: bool = True,
) -> dict[str, Any]:
    eval_envs = min(max(1, episodes), max(1, env_cfg.num_envs))
    env = SwitchingDuelVecEnv(replace(env_cfg, num_envs=eval_envs, seed=env_cfg.seed + 10_000))
    return evaluate_policy_in_env(policy, env, context_len, device, episodes, deterministic)


@torch.no_grad()
def evaluate_policy_in_env(
    policy: ActorCritic,
    env: Any,
    context_len: int,
    device: torch.device,
    episodes: int = 64,
    deterministic: bool = True,
) -> dict[str, Any]:
    obs_np = env.reset()
    eval_envs = env.num_envs
    obs = torch.nan_to_num(torch.as_tensor(obs_np, dtype=torch.float32, device=device), nan=0.0, posinf=1e6, neginf=-1e6)
    context = torch.zeros((eval_envs, context_len, env.obs_dim), dtype=torch.float32, device=device)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    rewards_by_age = {"age_0_3": [], "age_4_15": [], "age_16_plus": []}
    positive_reward_steps = 0
    reward_steps = 0
    switches = 0
    episode_length = int(getattr(env.cfg, "episode_length", 128))
    max_steps = max(episodes * episode_length * 2, episode_length)
    steps = 0

    policy.eval()
    while len(episode_returns) < episodes and steps < max_steps:
        action = policy.act(obs, context, deterministic=deterministic)
        next_obs_np, rewards_np, done_np, info = env.step(action.cpu().numpy())
        reward_steps += rewards_np.size
        positive_reward_steps += int(np.sum(rewards_np > 0.0))
        switches += int(np.sum(info["switched"]))
        ages = info["opponent_age"]
        rewards_by_age["age_0_3"].extend(rewards_np[ages <= 3].tolist())
        rewards_by_age["age_4_15"].extend(rewards_np[(ages >= 4) & (ages <= 15)].tolist())
        rewards_by_age["age_16_plus"].extend(rewards_np[ages >= 16].tolist())

        for ret, length in zip(info["episode_return"], info["episode_length"]):
            if np.isfinite(ret) and len(episode_returns) < episodes:
                episode_returns.append(float(ret))
                episode_lengths.append(int(length))

        obs = torch.nan_to_num(
            torch.as_tensor(next_obs_np, dtype=torch.float32, device=device),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        done = torch.as_tensor(done_np, dtype=torch.bool, device=device)
        context = append_context(context, obs, done)
        steps += 1

    episode_wins = int(np.sum(np.asarray(episode_returns, dtype=np.float32) > 0.0))
    episode_losses = int(np.sum(np.asarray(episode_returns, dtype=np.float32) < 0.0))
    episode_draws = len(episode_returns) - episode_wins - episode_losses
    episode_count = max(1, len(episode_returns))
    means = {
        f"eval_reward_{name}": float(np.mean(values)) if values else float("nan")
        for name, values in rewards_by_age.items()
    }
    means.update(
        {
            "eval_episodes": len(episode_returns),
            "eval_return_mean": float(np.mean(episode_returns)) if episode_returns else float("nan"),
            "eval_return_std": float(np.std(episode_returns)) if episode_returns else float("nan"),
            "eval_length_mean": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
            "eval_win_rate": episode_wins / episode_count,
            "eval_loss_rate": episode_losses / episode_count,
            "eval_draw_rate": episode_draws / episode_count,
            "eval_positive_step_rate": positive_reward_steps / max(1, reward_steps),
            "eval_switches": switches,
        }
    )
    policy.train()
    if hasattr(env, "close"):
        env.close()
    return means
