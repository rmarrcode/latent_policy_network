#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from latent_policy.evaluation import append_context
from latent_policy.melee_light_env import MeleeLightKnockbackEnv, load_melee_light_action_specs
from latent_policy.ppo import load_checkpoint
from latent_policy.utils import select_device


CHARACTER_IDS = {
    "marth": 0,
    "puff": 1,
    "fox": 2,
    "falco": 3,
    "falcon": 4,
}
DEFAULT_CHECKPOINT = Path("runs/melee_light_reward_signal_gpu/full_hyper_lvl0_post_reward_fix/checkpoint.pt")
DEFAULT_OUTPUT = Path("runs/melee_light_viewer/fox_vs_baseline_marth.gif")


def _character_id(value: str) -> int:
    lowered = value.lower()
    if lowered in CHARACTER_IDS:
        return CHARACTER_IDS[lowered]
    return int(value)


def _policy_obs(
    raw_obs: np.ndarray,
    reward: float,
    step_count: int,
    episode_length: int,
    target_dim: int,
) -> np.ndarray:
    raw = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
    if raw.size == target_dim:
        return raw
    extras = np.asarray([reward, step_count / max(1, episode_length)], dtype=np.float32)
    if raw.size + extras.size == target_dim:
        return np.concatenate([raw, extras])
    if raw.size < target_dim:
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[: raw.size] = raw
        end = min(target_dim, raw.size + extras.size)
        padded[raw.size:end] = extras[: end - raw.size]
        return padded
    return raw[:target_dim].astype(np.float32)


def _screenshot(env: MeleeLightKnockbackEnv, label: str) -> Image.Image:
    png = env._driver.get_screenshot_as_png()  # noqa: SLF001 - viewer needs the browser frame.
    image = Image.open(BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((12, 12, image.width - 12, 58), fill=(0, 0, 0, 155))
    draw.text((24, 26), label, fill=(255, 255, 255, 255))
    return image


def _save_gif(path: Path, frames: list[Image.Image], durations_ms: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations_ms,
        loop=0,
        optimize=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch/record a latent-policy Melee Light fight.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-out", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", help="Use argmax actions instead of sampling.")
    parser.add_argument("--show", action="store_true", help="Open a visible Chromium window while recording.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--final-hold-ms", type=int, default=900)
    parser.add_argument("--agent-character", type=_character_id, default=CHARACTER_IDS["fox"])
    parser.add_argument("--opponent-character", type=_character_id, default=CHARACTER_IDS["marth"])
    parser.add_argument("--opponent-level", type=int, default=3)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-episode-frames", type=int, default=240)
    parser.add_argument("--spawn-spacing", type=float, default=12.0)
    parser.add_argument("--spawn-y", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    policy, _, policy_cfg, ckpt = load_checkpoint(args.checkpoint, device)
    policy.to(device)
    policy.eval()
    action_specs = load_melee_light_action_specs()

    env = MeleeLightKnockbackEnv(
        frame_skip=args.frame_skip,
        max_episode_frames=args.max_episode_frames,
        agent_character=args.agent_character,
        opponent_character=args.opponent_character,
        opponent_level=args.opponent_level,
        close_spawn=True,
        spawn_spacing=args.spawn_spacing,
        spawn_y=args.spawn_y,
        headless=not args.show,
    )

    frames: list[Image.Image] = []
    durations: list[int] = []
    events: list[dict[str, Any]] = []
    step_duration_ms = max(1, int(1000 / max(args.fps, 0.1)))
    episode_length = max(1, int(np.ceil(args.max_episode_frames / max(1, args.frame_skip))))

    try:
        for episode in range(args.episodes):
            raw_obs, _ = env.reset(seed=args.seed + episode)
            obs_np = _policy_obs(raw_obs, 0.0, 0, episode_length, policy_cfg.obs_dim)
            obs = torch.as_tensor(obs_np[None, :], dtype=torch.float32, device=device)
            context = torch.zeros((1, policy_cfg.context_len, policy_cfg.obs_dim), dtype=torch.float32, device=device)
            frames.append(_screenshot(env, f"episode {episode + 1}: latent Fox vs baseline Marth"))
            durations.append(350)

            total_reward = 0.0
            for step in range(args.max_steps):
                with torch.no_grad():
                    action = int(policy.act(obs, context, deterministic=args.deterministic).item())
                raw_next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                total_reward += float(reward)
                action_name = action_specs[action]["name"] if 0 <= action < len(action_specs) else str(action)
                label = (
                    f"ep {episode + 1} step {step + 1}: Fox {action_name} | "
                    f"reward {reward:+.1f} | frame {info.get('frame_count', '?')}"
                )
                frames.append(_screenshot(env, label))
                durations.append(args.final_hold_ms if done else step_duration_ms)
                events.append(
                    {
                        "episode": episode + 1,
                        "step": step + 1,
                        "action": action,
                        "action_name": action_name,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info,
                    }
                )

                next_obs_np = _policy_obs(raw_next_obs, reward, step + 1, episode_length, policy_cfg.obs_dim)
                next_obs = torch.as_tensor(next_obs_np[None, :], dtype=torch.float32, device=device)
                done_tensor = torch.as_tensor([done], dtype=torch.bool, device=device)
                context = append_context(context, next_obs, done_tensor)
                obs = next_obs
                if args.show:
                    time.sleep(step_duration_ms / 1000)
                if done:
                    print(
                        {
                            "episode": episode + 1,
                            "return": total_reward,
                            "winner": info.get("winner"),
                            "loser": info.get("loser"),
                            "timeout": bool(info.get("timeout")),
                            "frames": info.get("frame_count"),
                        },
                        flush=True,
                    )
                    break
    finally:
        env.close()

    _save_gif(args.out, frames, durations)
    metadata_out = args.metadata_out or args.out.with_suffix(".json")
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_update": ckpt.get("update"),
                "device": str(device),
                "agent_character": args.agent_character,
                "opponent_character": args.opponent_character,
                "opponent_level": args.opponent_level,
                "deterministic": args.deterministic,
                "events": events,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print({"gif": str(args.out.resolve()), "metadata": str(metadata_out.resolve()), "frames": len(frames)})


if __name__ == "__main__":
    main()
