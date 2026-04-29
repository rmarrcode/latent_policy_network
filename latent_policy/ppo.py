from __future__ import annotations

import argparse
import copy
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml
from tqdm import tqdm

from latent_policy.envs import SwitchingDuelVecEnv, SwitchingGameConfig
from latent_policy.evaluation import append_context, evaluate_policy_in_env
from latent_policy.models import PolicyConfig, build_policy, count_parameters
from latent_policy.public_envs import PublicEnvConfig, build_public_env, clone_public_config
from latent_policy.utils import CSVLogger, explained_variance, make_run_dir, select_device, set_seed, write_json


@dataclass
class PolicyOptions:
    context_len: int = 16
    agent: str = "hyper_head"
    encoder: str = "gru"
    hidden_dim: int = 64
    latent_dim: int = 64
    generated_hidden_dim: int = 24
    attention_heads: int = 4
    attention_layers: int = 1
    weight_scale: float = 0.8


@dataclass
class TrainConfig:
    seed: int = 1
    device: str = "auto"
    run_dir: str = "runs"
    run_name: str | None = None
    total_updates: int = 80
    num_steps: int = 128
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.96
    gae_lambda: float = 0.92
    num_minibatches: int = 8
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.015
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.04
    eval_interval: int = 10
    eval_episodes: int = 64
    save_interval: int = 25
    torch_threads: int = 4
    progress: bool = True
    env: SwitchingGameConfig = field(default_factory=SwitchingGameConfig)
    public_env: PublicEnvConfig | None = None
    policy: PolicyOptions = field(default_factory=PolicyOptions)


def _update_dataclass(obj: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def config_from_dict(data: dict[str, Any]) -> TrainConfig:
    cfg = TrainConfig()
    data = dict(data)
    env_data = data.pop("env", {})
    public_env_data = data.pop("public_env", None)
    policy_data = data.pop("policy", {})
    _update_dataclass(cfg, data)
    _update_dataclass(cfg.env, env_data)
    if public_env_data is not None:
        cfg.public_env = PublicEnvConfig(**public_env_data)
    _update_dataclass(cfg.policy, policy_data)
    return cfg


def load_config(path: str | Path | None) -> TrainConfig:
    if path is None:
        return TrainConfig()
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return config_from_dict(data)


def make_env(cfg: TrainConfig, eval_envs: int | None = None, seed_offset: int = 0):
    if cfg.public_env is not None:
        public_cfg = copy.deepcopy(cfg.public_env)
        if eval_envs is not None:
            public_cfg = clone_public_config(public_cfg, num_envs=eval_envs)
        public_cfg.seed = cfg.seed + seed_offset
        return build_public_env(public_cfg)
    env_cfg = copy.deepcopy(cfg.env)
    env_cfg.seed = cfg.seed + seed_offset
    if eval_envs is not None:
        env_cfg.num_envs = eval_envs
    return SwitchingDuelVecEnv(env_cfg)


def make_policy_config(cfg: TrainConfig, env: Any) -> PolicyConfig:
    return PolicyConfig(
        obs_dim=env.obs_dim,
        action_dim=env.action_space_n,
        context_len=cfg.policy.context_len,
        agent=cfg.policy.agent,
        encoder=cfg.policy.encoder,
        hidden_dim=cfg.policy.hidden_dim,
        latent_dim=cfg.policy.latent_dim,
        generated_hidden_dim=cfg.policy.generated_hidden_dim,
        attention_heads=cfg.policy.attention_heads,
        attention_layers=cfg.policy.attention_layers,
        weight_scale=cfg.policy.weight_scale,
    )


def save_checkpoint(
    path: Path,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    policy_cfg: PolicyConfig,
    update: int,
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state": policy.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "policy_config": asdict(policy_cfg),
            "update": update,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, TrainConfig, PolicyConfig, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    cfg = config_from_dict(ckpt["config"])
    policy_cfg = PolicyConfig(**ckpt["policy_config"])
    policy = build_policy(policy_cfg).to(device)
    policy.load_state_dict(ckpt["model_state"])
    return policy, cfg, policy_cfg, ckpt


def train(cfg: TrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    torch.set_num_threads(max(1, cfg.torch_threads))
    device = select_device(cfg.device)
    if cfg.public_env is None:
        cfg.env.seed = cfg.seed
    else:
        cfg.public_env.seed = cfg.seed
    env = make_env(cfg)
    policy_cfg = make_policy_config(cfg, env)
    policy = build_policy(policy_cfg).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5)

    run_name = cfg.run_name or f"{cfg.policy.agent}_{cfg.policy.encoder}_seed{cfg.seed}"
    run_dir = make_run_dir(cfg.run_dir, run_name)
    logger = CSVLogger(run_dir / "metrics.csv")
    write_json(run_dir / "config.json", asdict(cfg))
    write_json(run_dir / "policy_config.json", asdict(policy_cfg))

    num_envs = env.num_envs
    obs_shape = (cfg.num_steps, num_envs, env.obs_dim)
    ctx_shape = (cfg.num_steps, num_envs, cfg.policy.context_len, env.obs_dim)
    obs_buf = torch.zeros(obs_shape, device=device)
    ctx_buf = torch.zeros(ctx_shape, device=device)
    actions_buf = torch.zeros((cfg.num_steps, num_envs), device=device, dtype=torch.long)
    logprobs_buf = torch.zeros((cfg.num_steps, num_envs), device=device)
    rewards_buf = torch.zeros((cfg.num_steps, num_envs), device=device)
    dones_buf = torch.zeros((cfg.num_steps, num_envs), device=device)
    values_buf = torch.zeros((cfg.num_steps, num_envs), device=device)

    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
    context = torch.zeros((num_envs, cfg.policy.context_len, env.obs_dim), dtype=torch.float32, device=device)
    global_step = 0
    recent_returns: deque[float] = deque(maxlen=128)
    final_metrics: dict[str, Any] = {}
    batch_size = cfg.num_steps * num_envs
    minibatch_size = max(1, batch_size // cfg.num_minibatches)
    indices = np.arange(batch_size)

    progress = tqdm(range(1, cfg.total_updates + 1), disable=not cfg.progress, desc="training")
    for update in progress:
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / max(1, cfg.total_updates)
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        age_rewards = {"age_0_3": [], "age_4_15": [], "age_16_plus": []}
        switches = 0
        for step in range(cfg.num_steps):
            global_step += num_envs
            obs_buf[step] = obs
            ctx_buf[step] = context

            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(obs, context)
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value

            next_obs_np, reward_np, done_np, info = env.step(action.cpu().numpy())
            rewards_buf[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
            dones_buf[step] = torch.as_tensor(done_np.astype(np.float32), dtype=torch.float32, device=device)

            ages = info["opponent_age"]
            age_rewards["age_0_3"].extend(reward_np[ages <= 3].tolist())
            age_rewards["age_4_15"].extend(reward_np[(ages >= 4) & (ages <= 15)].tolist())
            age_rewards["age_16_plus"].extend(reward_np[ages >= 16].tolist())
            switches += int(np.sum(info["switched"]))
            for ret in info["episode_return"]:
                if np.isfinite(ret):
                    recent_returns.append(float(ret))

            obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            done = torch.as_tensor(done_np, dtype=torch.bool, device=device)
            context = append_context(context, obs, done)

        with torch.no_grad():
            _, _, _, next_value = policy.get_action_and_value(obs, context)

        advantages = torch.zeros_like(rewards_buf, device=device)
        lastgaelam = torch.zeros(num_envs, device=device)
        for t in reversed(range(cfg.num_steps)):
            next_values = next_value if t == cfg.num_steps - 1 else values_buf[t + 1]
            next_nonterminal = 1.0 - dones_buf[t]
            delta = rewards_buf[t] + cfg.gamma * next_values * next_nonterminal - values_buf[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, env.obs_dim))
        b_ctx = ctx_buf.reshape((-1, cfg.policy.context_len, env.obs_dim))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        clipfracs = []
        approx_kl_value = 0.0
        for _ in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = indices[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    b_obs[mb_inds], b_ctx[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    approx_kl_value = float(approx_kl.item())
                    clipfracs.append(float(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()))

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not None and approx_kl_value > cfg.target_kl:
                break

        y_pred = b_values.detach()
        y_true = b_returns.detach()
        metrics: dict[str, Any] = {
            "update": update,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "policy_loss": float(pg_loss.item()),
            "value_loss": float(v_loss.item()),
            "entropy": float(entropy_loss.item()),
            "old_approx_kl": float(old_approx_kl.item()),
            "approx_kl": approx_kl_value,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "explained_variance": explained_variance(y_pred, y_true),
            "rollout_reward_mean": float(rewards_buf.mean().item()),
            "rollout_return_recent": float(np.mean(recent_returns)) if recent_returns else float("nan"),
            "rollout_switches": switches,
            "params": count_parameters(policy),
            "run_dir": str(run_dir),
        }
        for name, values in age_rewards.items():
            metrics[f"rollout_reward_{name}"] = float(np.mean(values)) if values else float("nan")

        if cfg.eval_interval > 0 and (update == 1 or update % cfg.eval_interval == 0 or update == cfg.total_updates):
            eval_envs = min(max(1, cfg.eval_episodes), max(1, env.num_envs))
            eval_env = make_env(cfg, eval_envs=eval_envs, seed_offset=10_000)
            eval_metrics = evaluate_policy_in_env(
                policy,
                eval_env,
                context_len=cfg.policy.context_len,
                device=device,
                episodes=cfg.eval_episodes,
                deterministic=True,
            )
            metrics.update(eval_metrics)

        logger.write(metrics)
        final_metrics = metrics
        progress.set_postfix(
            ret=f"{metrics.get('rollout_return_recent', float('nan')):.2f}",
            eval=f"{metrics.get('eval_return_mean', float('nan')):.2f}",
            rew=f"{metrics['rollout_reward_mean']:.3f}",
        )

        if cfg.save_interval > 0 and (update % cfg.save_interval == 0 or update == cfg.total_updates):
            save_checkpoint(run_dir / "checkpoint.pt", policy, optimizer, cfg, policy_cfg, update, metrics)

    write_json(run_dir / "final_metrics.json", final_metrics)
    final_metrics["run_dir"] = str(run_dir)
    return final_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train generated-weight RL policies.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--agent", choices=["static_mlp", "hyper_head", "film", "full_hyper"], default=None)
    parser.add_argument("--encoder", choices=["flat", "mean", "gru", "attention"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--public-kind", choices=["openspiel_matrix", "openspiel_turn", "pettingzoo_parallel", "gym_single"], default=None)
    parser.add_argument("--public-name", type=str, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.agent is not None:
        cfg.policy.agent = args.agent
    if args.encoder is not None:
        cfg.policy.encoder = args.encoder
    if args.seed is not None:
        cfg.seed = args.seed
    if args.total_updates is not None:
        cfg.total_updates = args.total_updates
    if args.num_envs is not None:
        if cfg.public_env is not None:
            cfg.public_env.num_envs = args.num_envs
        else:
            cfg.env.num_envs = args.num_envs
    if args.num_steps is not None:
        cfg.num_steps = args.num_steps
    if args.context_len is not None:
        cfg.policy.context_len = args.context_len
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.run_dir is not None:
        cfg.run_dir = args.run_dir
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.device is not None:
        cfg.device = args.device
    if args.public_kind is not None or args.public_name is not None:
        if cfg.public_env is None:
            cfg.public_env = PublicEnvConfig()
        if args.public_kind is not None:
            cfg.public_env.kind = args.public_kind
        if args.public_name is not None:
            cfg.public_env.name = args.public_name
    if args.episode_length is not None:
        if cfg.public_env is not None:
            cfg.public_env.episode_length = args.episode_length
        else:
            cfg.env.episode_length = args.episode_length
    if args.no_progress:
        cfg.progress = False
    return cfg


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    metrics = train(cfg)
    print(yaml.safe_dump(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
