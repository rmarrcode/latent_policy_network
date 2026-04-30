from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from latent_policy.evaluation import evaluate_policy_in_env


class DummyPolicy:
    def eval(self):
        pass

    def train(self):
        pass

    def act(self, obs: torch.Tensor, context: torch.Tensor, deterministic: bool) -> torch.Tensor:
        return torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)


class TerminalRewardEnv:
    num_envs = 1
    obs_dim = 1

    def __init__(self, returns: list[float]):
        self.cfg = SimpleNamespace(episode_length=1)
        self._returns = returns
        self._idx = 0

    def reset(self):
        return np.zeros((1, 1), dtype=np.float32)

    def step(self, actions):
        reward = float(self._returns[self._idx])
        self._idx += 1
        info = {
            "switched": np.array([False]),
            "opponent_age": np.array([0]),
            "episode_return": np.array([reward], dtype=np.float32),
            "episode_length": np.array([1], dtype=np.int32),
        }
        return np.zeros((1, 1), dtype=np.float32), np.array([reward], dtype=np.float32), np.array([True]), info

    def close(self):
        pass


def test_eval_win_rate_uses_completed_episodes_not_steps():
    env = TerminalRewardEnv([1.0, -1.0, 0.0])
    metrics = evaluate_policy_in_env(DummyPolicy(), env, context_len=2, device=torch.device("cpu"), episodes=3)

    assert metrics["eval_win_rate"] == 1 / 3
    assert metrics["eval_loss_rate"] == 1 / 3
    assert metrics["eval_draw_rate"] == 1 / 3
    assert metrics["eval_positive_step_rate"] == 1 / 3
