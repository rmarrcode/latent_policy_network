from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SwitchingGameConfig:
    """Configuration for a vectorized hidden-opponent repeated duel.

    The game is intentionally small enough for rapid research iteration but
    still forces adaptation: the opponent's hidden style and parameters switch
    within an episode, so stale policies are punished.
    """

    num_envs: int = 64
    n_actions: int = 5
    episode_length: int = 128
    switch_hazard: float = 0.035
    min_switch_interval: int = 8
    opponent_noise: float = 0.14
    count_decay: float = 0.92
    draw_reward: float = -0.03
    lose_reward: float = -1.0
    win_reward: float = 1.0
    seed: int = 0


OPPONENT_TYPES = (
    "fixed",
    "mirror_last",
    "counter_last",
    "cycle",
    "exploit_frequency",
    "noisy_uniform",
)


class SwitchingDuelVecEnv:
    """Vectorized repeated 1v1 cyclic dominance game with hidden switches.

    Action i beats the previous half of actions on a cycle. With five actions,
    action 2 beats actions 0 and 1, loses to actions 3 and 4, and draws with 2.
    Opponents have hidden strategies such as fixed bias, mirroring, countering
    the agent's last action, cycling, or exploiting recent action frequencies.
    """

    def __init__(self, cfg: SwitchingGameConfig):
        if cfg.n_actions % 2 == 0:
            raise ValueError("n_actions must be odd for cyclic dominance payoffs")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.num_envs = cfg.num_envs
        self.n_actions = cfg.n_actions
        self.obs_dim = 2 * cfg.n_actions + 2
        self.action_space_n = cfg.n_actions

        self.step_count = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self.last_agent_action = np.full(self.num_envs, -1, dtype=np.int64)
        self.last_opp_action = np.full(self.num_envs, -1, dtype=np.int64)
        self.last_reward = np.zeros(self.num_envs, dtype=np.float32)
        self.opponent_type = np.zeros(self.num_envs, dtype=np.int64)
        self.opponent_param = np.zeros(self.num_envs, dtype=np.int64)
        self.opponent_phase = np.zeros(self.num_envs, dtype=np.int64)
        self.steps_since_switch = np.zeros(self.num_envs, dtype=np.int32)
        self.agent_action_counts = np.ones(
            (self.num_envs, self.n_actions), dtype=np.float32
        )

        self.reset()

    def reset(self, indices: np.ndarray | list[int] | None = None) -> np.ndarray:
        if indices is None:
            idx = np.arange(self.num_envs)
        else:
            idx = np.asarray(indices, dtype=np.int64)
        self.step_count[idx] = 0
        self.episode_return[idx] = 0.0
        self.last_agent_action[idx] = -1
        self.last_opp_action[idx] = -1
        self.last_reward[idx] = 0.0
        self.agent_action_counts[idx] = 1.0
        self._sample_opponents(idx, avoid_current=False)
        return self._obs()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape {(self.num_envs,)}, got {actions.shape}")
        if np.any(actions < 0) or np.any(actions >= self.n_actions):
            raise ValueError("actions out of range")

        switched = self._maybe_switch_opponents()
        age_at_action = self.steps_since_switch.copy()
        opp_actions = self._opponent_actions(actions)
        rewards = self._payoff(actions, opp_actions).astype(np.float32)

        self._update_counts(actions)
        self.last_agent_action = actions.copy()
        self.last_opp_action = opp_actions.copy()
        self.last_reward = rewards.copy()
        self.step_count += 1
        self.steps_since_switch += 1
        self.episode_return += rewards

        done = self.step_count >= self.cfg.episode_length
        final_return = np.where(done, self.episode_return, np.nan).astype(np.float32)
        final_length = np.where(done, self.step_count, 0).astype(np.int32)
        info: dict[str, Any] = {
            "switched": switched,
            "opponent_age": age_at_action,
            "opponent_type": self.opponent_type.copy(),
            "opponent_name": np.asarray([OPPONENT_TYPES[i] for i in self.opponent_type]),
            "episode_return": final_return,
            "episode_length": final_length,
        }

        if np.any(done):
            self.reset(np.nonzero(done)[0])

        return self._obs(), rewards, done.astype(np.bool_), info

    def _sample_opponents(self, idx: np.ndarray, avoid_current: bool) -> None:
        if idx.size == 0:
            return
        new_types = self.rng.integers(0, len(OPPONENT_TYPES), size=idx.size)
        if avoid_current:
            same = new_types == self.opponent_type[idx]
            new_types[same] = (new_types[same] + self.rng.integers(1, len(OPPONENT_TYPES), size=np.sum(same))) % len(OPPONENT_TYPES)
        self.opponent_type[idx] = new_types
        self.opponent_param[idx] = self.rng.integers(0, self.n_actions, size=idx.size)
        self.opponent_phase[idx] = self.rng.integers(2, 8, size=idx.size)
        self.steps_since_switch[idx] = 0

    def _maybe_switch_opponents(self) -> np.ndarray:
        eligible = self.steps_since_switch >= self.cfg.min_switch_interval
        random_switch = self.rng.random(self.num_envs) < self.cfg.switch_hazard
        switched = eligible & random_switch
        if np.any(switched):
            self._sample_opponents(np.nonzero(switched)[0], avoid_current=True)
        return switched

    def _opponent_actions(self, agent_actions: np.ndarray) -> np.ndarray:
        preferred = np.zeros(self.num_envs, dtype=np.int64)
        last_agent = np.where(self.last_agent_action >= 0, self.last_agent_action, self.opponent_param)

        for env_id in range(self.num_envs):
            typ = OPPONENT_TYPES[self.opponent_type[env_id]]
            param = self.opponent_param[env_id]
            if typ == "fixed":
                preferred[env_id] = param
            elif typ == "mirror_last":
                preferred[env_id] = last_agent[env_id]
            elif typ == "counter_last":
                preferred[env_id] = self._best_response(last_agent[env_id])
            elif typ == "cycle":
                phase = max(1, int(self.opponent_phase[env_id]))
                preferred[env_id] = (param + self.steps_since_switch[env_id] // phase) % self.n_actions
            elif typ == "exploit_frequency":
                common = int(np.argmax(self.agent_action_counts[env_id]))
                preferred[env_id] = self._best_response(common)
            elif typ == "noisy_uniform":
                preferred[env_id] = self.rng.integers(0, self.n_actions)
            else:
                raise RuntimeError(f"unknown opponent type: {typ}")

        noise = self.rng.random(self.num_envs) < self.cfg.opponent_noise
        preferred[noise] = self.rng.integers(0, self.n_actions, size=np.sum(noise))
        return preferred

    def _best_response(self, action: int) -> int:
        return (int(action) + 1) % self.n_actions

    def _payoff(self, agent_actions: np.ndarray, opp_actions: np.ndarray) -> np.ndarray:
        diff = (agent_actions - opp_actions) % self.n_actions
        rewards = np.full(self.num_envs, self.cfg.lose_reward, dtype=np.float32)
        rewards[diff == 0] = self.cfg.draw_reward
        rewards[(diff >= 1) & (diff <= self.n_actions // 2)] = self.cfg.win_reward
        return rewards

    def _update_counts(self, actions: np.ndarray) -> None:
        self.agent_action_counts *= self.cfg.count_decay
        self.agent_action_counts[np.arange(self.num_envs), actions] += 1.0

    def _obs(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        agent_seen = self.last_agent_action >= 0
        opp_seen = self.last_opp_action >= 0
        obs[np.nonzero(agent_seen)[0], self.last_agent_action[agent_seen]] = 1.0
        offset = self.n_actions
        obs[np.nonzero(opp_seen)[0], offset + self.last_opp_action[opp_seen]] = 1.0
        obs[:, 2 * self.n_actions] = self.last_reward
        obs[:, 2 * self.n_actions + 1] = self.step_count / max(1, self.cfg.episode_length)
        return obs
