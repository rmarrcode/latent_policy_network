from __future__ import annotations

import importlib
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np


@dataclass
class PublicEnvConfig:
    kind: str = "openspiel_matrix"
    name: str = "matrix_rps"
    num_envs: int = 32
    episode_length: int = 64
    seed: int = 0
    max_cycles: int = 64
    train_agent: str | None = None
    switch_hazard: float = 0.05
    min_switch_interval: int = 8
    opponent_pool: tuple[str, ...] = (
        "random",
        "fixed0",
        "fixed1",
        "cycle",
        "repeat_agent",
        "beat_last",
    )
    env_kwargs: dict[str, Any] = field(default_factory=dict)


def build_public_env(cfg: PublicEnvConfig):
    if cfg.kind == "openspiel_matrix":
        return OpenSpielMatrixVecEnv(cfg)
    if cfg.kind == "openspiel_turn":
        return OpenSpielTurnBasedVecEnv(cfg)
    if cfg.kind == "pettingzoo_parallel":
        return PettingZooParallelDiscreteVecEnv(cfg)
    if cfg.kind == "gym_single":
        return GymSingleDiscreteVecEnv(cfg)
    raise ValueError(f"unknown public env kind: {cfg.kind}")


def clone_public_config(cfg: PublicEnvConfig, num_envs: int | None = None, seed: int | None = None) -> PublicEnvConfig:
    kwargs: dict[str, Any] = {}
    if num_envs is not None:
        kwargs["num_envs"] = num_envs
    if seed is not None:
        kwargs["seed"] = seed
    return replace(cfg, **kwargs)


def _discrete_obs(value: Any, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    out[int(value)] = 1.0
    return out


def _flatten_obs(value: Any, space: Any | None = None) -> np.ndarray:
    if space is not None and space.__class__.__name__ == "Discrete":
        return _discrete_obs(value, int(space.n))
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == ():
        return arr.reshape(1)
    return np.nan_to_num(arr.reshape(-1).astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)


def _sample_from_legal(rng: np.random.Generator, legal: list[int] | np.ndarray) -> int:
    legal = list(map(int, legal))
    return int(legal[int(rng.integers(0, len(legal)))])


def _scripted_action(
    policy: str,
    n_actions: int,
    rng: np.random.Generator,
    legal: list[int] | np.ndarray | None,
    last_agent_action: int,
    last_opp_action: int,
    step: int,
    agent_action_counts: np.ndarray | None = None,
    opp_action_counts: np.ndarray | None = None,
) -> int:
    legal_list = list(range(n_actions)) if legal is None else list(map(int, legal))
    if not legal_list:
        return 0

    def _count_choice(counts: np.ndarray | None, choose_max: bool) -> int | None:
        if counts is None:
            return None
        legal_counts = [(action, int(counts[action]) if action < len(counts) else 0) for action in legal_list]
        if not legal_counts or all(count == 0 for _, count in legal_counts):
            return None
        target = max(count for _, count in legal_counts) if choose_max else min(count for _, count in legal_counts)
        for action, count in legal_counts:
            if count == target:
                return int(action)
        return None

    if policy == "random":
        return _sample_from_legal(rng, legal_list)
    if policy == "fixed0":
        return legal_list[0]
    if policy == "fixed1":
        return legal_list[min(1, len(legal_list) - 1)]
    if policy == "fixed_last":
        return legal_list[-1]
    if policy == "cycle":
        return legal_list[step % len(legal_list)]
    if policy == "repeat_agent" and last_agent_action in legal_list:
        return int(last_agent_action)
    if policy == "beat_last":
        candidate = (max(0, last_agent_action) + 1) % max(1, n_actions)
        if candidate in legal_list:
            return int(candidate)
    if policy == "majority_agent":
        candidate = _count_choice(agent_action_counts, choose_max=True)
        if candidate is not None:
            return candidate
    if policy == "minority_agent":
        candidate = _count_choice(agent_action_counts, choose_max=False)
        if candidate is not None:
            return candidate
    if policy == "counter_majority":
        anchor = _count_choice(agent_action_counts, choose_max=True)
        if anchor is not None:
            for offset in range(1, max(2, n_actions + 1)):
                candidate = (anchor + offset) % max(1, n_actions)
                if candidate in legal_list:
                    return int(candidate)
    if policy == "majority_self":
        candidate = _count_choice(opp_action_counts, choose_max=True)
        if candidate is not None:
            return candidate
    if policy == "repeat_self" and last_opp_action in legal_list:
        return int(last_opp_action)
    return _sample_from_legal(rng, legal_list)


MELEE_LIGHT_CHARACTER_NAMES = {
    0: "marth",
    1: "puff",
    2: "fox",
    3: "falco",
    4: "falcon",
}


def _parse_int_pool(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return (int(value),)
    if isinstance(value, str):
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    return tuple(int(item) for item in value)


def _melee_action(name: str, n_actions: int) -> int:
    names = {
        "idle": 0,
        "left": 1,
        "right": 2,
        "up": 3,
        "down": 4,
        "jump": 5,
        "shield": 6,
        "jab": 7,
        "tilt_up": 8,
        "tilt_down": 9,
        "tilt_left": 10,
        "tilt_right": 11,
        "smash_up": 12,
        "smash_down": 13,
        "smash_left": 14,
        "smash_right": 15,
        "special_n": 16,
        "special_up": 17,
        "special_left": 18,
        "special_right": 19,
    }
    return min(max(0, names[name]), max(0, n_actions - 1))


def _mirror_melee_action(action: int) -> int:
    mirror = {
        1: 2,
        2: 1,
        10: 11,
        11: 10,
        14: 15,
        15: 14,
        18: 19,
        19: 18,
    }
    return mirror.get(int(action), int(action))


def _melee_light_scripted_action(
    policy: str,
    n_actions: int,
    rng: np.random.Generator,
    obs: np.ndarray,
    agent_action: int,
    last_agent_action: int,
    last_opp_action: int,
    step: int,
    agent_action_counts: np.ndarray | None,
) -> int:
    raw = np.asarray(obs, dtype=np.float32).reshape(-1)
    if raw.size < 30:
        return _scripted_action(policy, n_actions, rng, None, last_agent_action, last_opp_action, step, agent_action_counts)

    target_dx = -float(raw[28])
    distance = abs(target_dx)
    grounded = bool(raw[21] > 0.5)
    toward = _melee_action("left" if target_dx < 0 else "right", n_actions)
    away = _melee_action("right" if target_dx < 0 else "left", n_actions)
    tilt_toward = _melee_action("tilt_left" if target_dx < 0 else "tilt_right", n_actions)
    smash_toward = _melee_action("smash_left" if target_dx < 0 else "smash_right", n_actions)
    special_toward = _melee_action("special_left" if target_dx < 0 else "special_right", n_actions)

    def choice(names: list[str]) -> int:
        return _melee_action(names[int(rng.integers(0, len(names)))], n_actions)

    if policy in {"random", "noisy"}:
        return int(rng.integers(0, n_actions))
    if policy in {"idle", "fixed0"}:
        return _melee_action("idle", n_actions)
    if policy == "mirror_agent":
        return _mirror_melee_action(agent_action)
    if policy == "rushdown":
        if distance > 18:
            return _melee_action("jump", n_actions) if grounded and step % 5 == 0 else toward
        return smash_toward if distance < 9 and step % 3 == 0 else choice(["jab", "tilt_up"])
    if policy == "approach_jab":
        if distance > 13:
            return toward
        return choice(["jab", "tilt_up", "tilt_down"])
    if policy == "spacer":
        if distance < 11:
            return away
        if distance > 24:
            return toward
        return tilt_toward
    if policy == "zoner":
        if distance < 16:
            return choice(["shield", "jump"]) if grounded and step % 2 == 0 else away
        return _melee_action("special_n", n_actions) if step % 3 else special_toward
    if policy == "counter_poke":
        if distance < 10 and step % 4 == 0:
            return _melee_action("shield", n_actions)
        if distance < 18:
            return tilt_toward
        return toward if step % 2 else _melee_action("idle", n_actions)
    if policy == "jumper":
        if grounded and step % 3 == 0:
            return _melee_action("jump", n_actions)
        return choice(["tilt_up", "smash_up", "special_up"]) if distance < 14 else toward
    if policy == "anti_frequency" and agent_action_counts is not None:
        common = int(np.argmax(agent_action_counts))
        if common in {1, 2}:
            return smash_toward
        if common in {6, 0}:
            return toward
        return _melee_action("shield", n_actions) if distance < 12 else special_toward
    return _scripted_action(policy, n_actions, rng, None, last_agent_action, last_opp_action, step, agent_action_counts)


class BasePublicVecEnv:
    cfg: PublicEnvConfig
    num_envs: int
    obs_dim: int
    action_space_n: int

    def _init_common(self, cfg: PublicEnvConfig) -> None:
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.rng = np.random.default_rng(cfg.seed)
        self.step_count = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self.last_reward = np.zeros(self.num_envs, dtype=np.float32)
        self.steps_since_switch = np.zeros(self.num_envs, dtype=np.int32)
        self.opponent_idx = np.zeros(self.num_envs, dtype=np.int64)
        self.last_agent_action = np.full(self.num_envs, -1, dtype=np.int64)
        self.last_opp_action = np.full(self.num_envs, -1, dtype=np.int64)
        self.agent_action_counts: np.ndarray | None = None
        self.opp_action_counts: np.ndarray | None = None

    def _init_action_stats(self, n_actions: int) -> None:
        self.agent_action_counts = np.zeros((self.num_envs, n_actions), dtype=np.int32)
        self.opp_action_counts = np.zeros((self.num_envs, n_actions), dtype=np.int32)

    def _reset_action_stats(self, idx: np.ndarray) -> None:
        if self.agent_action_counts is not None:
            self.agent_action_counts[idx] = 0
        if self.opp_action_counts is not None:
            self.opp_action_counts[idx] = 0

    def _record_actions(self, agent_actions: np.ndarray, opp_actions: np.ndarray | None = None) -> None:
        if self.agent_action_counts is not None:
            agent_actions = np.asarray(agent_actions, dtype=np.int64)
            valid_agent = (agent_actions >= 0) & (agent_actions < self.agent_action_counts.shape[1])
            if np.any(valid_agent):
                idx = np.nonzero(valid_agent)[0]
                self.agent_action_counts[idx, agent_actions[valid_agent]] += 1
        if opp_actions is not None and self.opp_action_counts is not None:
            opp_actions = np.asarray(opp_actions, dtype=np.int64)
            valid_opp = (opp_actions >= 0) & (opp_actions < self.opp_action_counts.shape[1])
            if np.any(valid_opp):
                idx = np.nonzero(valid_opp)[0]
                self.opp_action_counts[idx, opp_actions[valid_opp]] += 1

    def _sample_opponents(self, idx: np.ndarray) -> None:
        if idx.size == 0:
            return
        pool_size = max(1, len(self.cfg.opponent_pool))
        self.opponent_idx[idx] = self.rng.integers(0, pool_size, size=idx.size)
        self.steps_since_switch[idx] = 0

    def _maybe_switch(self) -> np.ndarray:
        eligible = self.steps_since_switch >= self.cfg.min_switch_interval
        switched = eligible & (self.rng.random(self.num_envs) < self.cfg.switch_hazard)
        if np.any(switched):
            self._sample_opponents(np.nonzero(switched)[0])
        return switched

    def _opponent_policy(self, env_id: int) -> str:
        return self.cfg.opponent_pool[int(self.opponent_idx[env_id]) % len(self.cfg.opponent_pool)]

    def close(self) -> None:
        return None


class OpenSpielMatrixVecEnv(BasePublicVecEnv):
    def __init__(self, cfg: PublicEnvConfig):
        import pyspiel

        self._init_common(cfg)
        self.game = pyspiel.load_game(cfg.name)
        state = self.game.new_initial_state()
        if not state.is_simultaneous_node():
            raise ValueError(f"{cfg.name} is not a simultaneous matrix game")
        self.action_space_n = int(self.game.num_distinct_actions())
        self._init_action_stats(self.action_space_n)
        self.obs_dim = 2 * self.action_space_n + 2
        self.reset()

    def reset(self, indices: np.ndarray | list[int] | None = None) -> np.ndarray:
        idx = np.arange(self.num_envs) if indices is None else np.asarray(indices, dtype=np.int64)
        self.step_count[idx] = 0
        self.episode_return[idx] = 0.0
        self.last_reward[idx] = 0.0
        self.last_agent_action[idx] = -1
        self.last_opp_action[idx] = -1
        self._reset_action_stats(idx)
        self._sample_opponents(idx)
        return self._obs()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.int64) % self.action_space_n
        switched = self._maybe_switch()
        age_at_action = self.steps_since_switch.copy()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        opp_actions = np.zeros(self.num_envs, dtype=np.int64)

        for env_id in range(self.num_envs):
            policy = self._opponent_policy(env_id)
            opp_action = _scripted_action(
                policy,
                self.action_space_n,
                self.rng,
                None,
                int(self.last_agent_action[env_id]),
                int(self.last_opp_action[env_id]),
                int(self.step_count[env_id]),
                None if self.agent_action_counts is None else self.agent_action_counts[env_id],
                None if self.opp_action_counts is None else self.opp_action_counts[env_id],
            )
            state = self.game.new_initial_state()
            state.apply_actions([int(actions[env_id]), int(opp_action)])
            rewards[env_id] = float(state.returns()[0])
            opp_actions[env_id] = opp_action

        self._record_actions(actions, opp_actions)
        self.last_agent_action = actions.copy()
        self.last_opp_action = opp_actions
        self.last_reward = rewards.copy()
        self.step_count += 1
        self.steps_since_switch += 1
        self.episode_return += rewards
        done = self.step_count >= self.cfg.episode_length
        info = self._info(done, switched, age_at_action)
        if np.any(done):
            self.reset(np.nonzero(done)[0])
        return self._obs(), rewards, done.astype(np.bool_), info

    def _info(self, done: np.ndarray, switched: np.ndarray, age: np.ndarray) -> dict[str, Any]:
        return {
            "switched": switched,
            "opponent_age": age,
            "opponent_type": self.opponent_idx.copy(),
            "opponent_name": np.asarray([self._opponent_policy(i) for i in range(self.num_envs)]),
            "episode_return": np.where(done, self.episode_return, np.nan).astype(np.float32),
            "episode_length": np.where(done, self.step_count, 0).astype(np.int32),
        }

    def _obs(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        seen_agent = self.last_agent_action >= 0
        seen_opp = self.last_opp_action >= 0
        obs[np.nonzero(seen_agent)[0], self.last_agent_action[seen_agent]] = 1.0
        offset = self.action_space_n
        obs[np.nonzero(seen_opp)[0], offset + self.last_opp_action[seen_opp]] = 1.0
        obs[:, -2] = self.last_reward
        obs[:, -1] = self.step_count / max(1, self.cfg.episode_length)
        return obs


class OpenSpielTurnBasedVecEnv(BasePublicVecEnv):
    def __init__(self, cfg: PublicEnvConfig):
        import pyspiel

        self._init_common(cfg)
        self.pyspiel = pyspiel
        self.game = pyspiel.load_game(cfg.name)
        self.action_space_n = int(self.game.num_distinct_actions())
        self._init_action_stats(self.action_space_n)
        self.states = [self.game.new_initial_state() for _ in range(self.num_envs)]
        self.prev_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._obs_extra_dim = 2
        self._use_info_state = False
        self.reset()
        if cfg.name not in {"tic_tac_toe", "connect_four", "ultimate_tic_tac_toe"}:
            self._use_info_state = self._detect_info_state()
        self.obs_dim = self._state_obs(0).shape[0] + self._obs_extra_dim

    def reset(self, indices: np.ndarray | list[int] | None = None) -> np.ndarray:
        idx = np.arange(self.num_envs) if indices is None else np.asarray(indices, dtype=np.int64)
        self.step_count[idx] = 0
        self.episode_return[idx] = 0.0
        self.last_reward[idx] = 0.0
        self.last_agent_action[idx] = -1
        self.last_opp_action[idx] = -1
        self._reset_action_stats(idx)
        self._sample_opponents(idx)
        for env_id in idx:
            self.states[int(env_id)] = self.game.new_initial_state()
            self.prev_returns[int(env_id)] = 0.0
            self._advance_to_agent(int(env_id))
        return self._obs()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.int64)
        switched = self._maybe_switch()
        age_at_action = self.steps_since_switch.copy()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=np.bool_)
        agent_actions_taken = np.full(self.num_envs, -1, dtype=np.int64)
        opp_actions_taken = np.full(self.num_envs, -1, dtype=np.int64)

        for env_id in range(self.num_envs):
            state = self.states[env_id]
            if state.is_terminal():
                done[env_id] = True
                continue
            if state.current_player() != 0:
                opp_actions_taken[env_id] = self._advance_to_agent(env_id)
                state = self.states[env_id]
            if not state.is_terminal():
                legal = state.legal_actions(0)
                action = int(actions[env_id])
                if action not in legal:
                    action = legal[action % len(legal)]
                state.apply_action(action)
                self.last_agent_action[env_id] = action
                agent_actions_taken[env_id] = action
                next_opp_action = self._advance_to_agent(env_id)
                if next_opp_action >= 0:
                    opp_actions_taken[env_id] = next_opp_action

            returns0 = float(self.states[env_id].returns()[0])
            rewards[env_id] = returns0 - float(self.prev_returns[env_id])
            self.prev_returns[env_id] = returns0
            done[env_id] = self.states[env_id].is_terminal()

        self._record_actions(agent_actions_taken, opp_actions_taken)
        self.last_reward = rewards.copy()
        self.step_count += 1
        self.steps_since_switch += 1
        self.episode_return += rewards
        forced_done = self.step_count >= self.cfg.episode_length
        done = done | forced_done
        info = {
            "switched": switched,
            "opponent_age": age_at_action,
            "opponent_type": self.opponent_idx.copy(),
            "opponent_name": np.asarray([self._opponent_policy(i) for i in range(self.num_envs)]),
            "episode_return": np.where(done, self.episode_return, np.nan).astype(np.float32),
            "episode_length": np.where(done, self.step_count, 0).astype(np.int32),
        }
        if np.any(done):
            self.reset(np.nonzero(done)[0])
        return self._obs(), rewards, done.astype(np.bool_), info

    def _advance_to_agent(self, env_id: int) -> int:
        state = self.states[env_id]
        guard = 0
        last_opp_action = -1
        while not state.is_terminal() and state.current_player() != 0:
            guard += 1
            if guard > 512:
                raise RuntimeError(f"advance guard tripped for {self.cfg.name}")
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                state.apply_action(int(self.rng.choice(actions, p=np.asarray(probs, dtype=np.float64))))
                continue
            player = state.current_player()
            legal = state.legal_actions(player)
            policy = self._opponent_policy(env_id)
            opp_action = _scripted_action(
                policy,
                self.action_space_n,
                self.rng,
                legal,
                int(self.last_agent_action[env_id]),
                int(self.last_opp_action[env_id]),
                int(self.step_count[env_id]),
                None if self.agent_action_counts is None else self.agent_action_counts[env_id],
                None if self.opp_action_counts is None else self.opp_action_counts[env_id],
            )
            self.last_opp_action[env_id] = opp_action
            last_opp_action = opp_action
            state.apply_action(opp_action)
        return last_opp_action

    def _detect_info_state(self) -> bool:
        try:
            self.states[0].information_state_tensor(0)
            return True
        except Exception:
            return False

    def _state_obs(self, env_id: int) -> np.ndarray:
        state = self.states[env_id]
        if state.is_terminal():
            state = self.game.new_initial_state()
        if self._use_info_state:
            return np.asarray(state.information_state_tensor(0), dtype=np.float32).reshape(-1)
        return np.asarray(state.observation_tensor(0), dtype=np.float32).reshape(-1)

    def _obs(self) -> np.ndarray:
        base = np.stack([self._state_obs(i) for i in range(self.num_envs)], axis=0)
        extra = np.zeros((self.num_envs, self._obs_extra_dim), dtype=np.float32)
        extra[:, 0] = self.last_reward
        extra[:, 1] = self.step_count / max(1, self.cfg.episode_length)
        return np.concatenate([base.astype(np.float32), extra], axis=1)


class PettingZooParallelDiscreteVecEnv(BasePublicVecEnv):
    def __init__(self, cfg: PublicEnvConfig):
        self._init_common(cfg)
        module_name, _, factory_name = cfg.name.partition(":")
        self.module = importlib.import_module(module_name)
        self.factory_name = factory_name or "parallel_env"
        self.envs = [self._make_env(i) for i in range(self.num_envs)]
        initial_obs = []
        self.train_agents = []
        for i, env in enumerate(self.envs):
            out = env.reset(seed=cfg.seed + i)
            obs = out[0] if isinstance(out, tuple) else out
            initial_obs.append(obs)
            self.train_agents.append(self._select_train_agent(env, obs))
        train_agent = self.train_agents[0]
        self.action_space_n = int(self.envs[0].action_space(train_agent).n)
        self._init_action_stats(self.action_space_n)
        self._obs_space = self.envs[0].observation_space(train_agent)
        self._last_obs = np.zeros((self.num_envs, _flatten_obs(initial_obs[0][train_agent], self._obs_space).shape[0]), dtype=np.float32)
        for i, obs in enumerate(initial_obs):
            agent = self.train_agents[i]
            self._last_obs[i] = _flatten_obs(obs[agent], self.envs[i].observation_space(agent))
        self.step_count[:] = 0
        self.episode_return[:] = 0.0
        self.last_reward[:] = 0.0
        self.last_agent_action[:] = -1
        self.last_opp_action[:] = -1
        self._sample_opponents(np.arange(self.num_envs))
        self.obs_dim = self._last_obs.shape[1] + 2

    def _make_env(self, idx: int):
        factory: Callable[..., Any] = getattr(self.module, self.factory_name)
        kwargs: dict[str, Any] = {"render_mode": None}
        if "mpe2." in self.cfg.name:
            kwargs["max_cycles"] = self.cfg.max_cycles
        elif "magent2." in self.cfg.name:
            kwargs["max_cycles"] = self.cfg.max_cycles
            kwargs["map_size"] = 12
        elif "pettingzoo.classic.rps" in self.cfg.name:
            kwargs["num_actions"] = 3
            kwargs["max_cycles"] = self.cfg.max_cycles
        kwargs.update(self.cfg.env_kwargs)
        env = factory(**kwargs)
        return env

    def _select_train_agent(self, env: Any, obs: dict[str, Any] | None) -> str:
        agents = list(obs.keys()) if obs is not None else list(getattr(env, "agents", getattr(env, "possible_agents", [])))
        if self.cfg.train_agent is not None and self.cfg.train_agent in agents:
            return self.cfg.train_agent
        return agents[0]

    def reset(self, indices: np.ndarray | list[int] | None = None) -> np.ndarray:
        idx = np.arange(self.num_envs) if indices is None else np.asarray(indices, dtype=np.int64)
        for env_id in idx:
            out = self.envs[int(env_id)].reset(seed=self.cfg.seed + int(env_id) * 997 + int(self.step_count[int(env_id)]))
            obs = out[0] if isinstance(out, tuple) else out
            self.train_agents[int(env_id)] = self._select_train_agent(self.envs[int(env_id)], obs)
            agent = self.train_agents[int(env_id)]
            self._last_obs[int(env_id)] = _flatten_obs(obs[agent], self.envs[int(env_id)].observation_space(agent))
        self.step_count[idx] = 0
        self.episode_return[idx] = 0.0
        self.last_reward[idx] = 0.0
        self.last_agent_action[idx] = -1
        self.last_opp_action[idx] = -1
        self._reset_action_stats(idx)
        self._sample_opponents(idx)
        return self._obs()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.int64)
        switched = self._maybe_switch()
        age_at_action = self.steps_since_switch.copy()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=np.bool_)
        opp_actions = np.full(self.num_envs, -1, dtype=np.int64)

        for env_id, env in enumerate(self.envs):
            train_agent = self.train_agents[env_id]
            if train_agent not in env.agents:
                done[env_id] = True
                continue
            action_dict: dict[str, int] = {}
            for agent in env.agents:
                n = int(env.action_space(agent).n)
                if agent == train_agent:
                    action = int(actions[env_id] % n)
                    self.last_agent_action[env_id] = action
                else:
                    action = _scripted_action(
                        self._opponent_policy(env_id),
                        n,
                        self.rng,
                        None,
                        int(self.last_agent_action[env_id]),
                        int(self.last_opp_action[env_id]),
                        int(self.step_count[env_id]),
                        None if self.agent_action_counts is None else self.agent_action_counts[env_id],
                        None if self.opp_action_counts is None else self.opp_action_counts[env_id],
                    )
                    self.last_opp_action[env_id] = action
                    opp_actions[env_id] = action
                action_dict[agent] = action

            obs, reward_dict, terminations, truncations, _ = env.step(action_dict)
            rewards[env_id] = float(reward_dict.get(train_agent, 0.0))
            done[env_id] = bool(terminations.get(train_agent, False) or truncations.get(train_agent, False) or train_agent not in obs)
            if not done[env_id]:
                self._last_obs[env_id] = _flatten_obs(obs[train_agent], env.observation_space(train_agent))

        self._record_actions(actions, opp_actions)
        self.last_reward = rewards.copy()
        self.step_count += 1
        self.steps_since_switch += 1
        self.episode_return += rewards
        forced_done = self.step_count >= self.cfg.episode_length
        done = done | forced_done
        info = {
            "switched": switched,
            "opponent_age": age_at_action,
            "opponent_type": self.opponent_idx.copy(),
            "opponent_name": np.asarray([self._opponent_policy(i) for i in range(self.num_envs)]),
            "episode_return": np.where(done, self.episode_return, np.nan).astype(np.float32),
            "episode_length": np.where(done, self.step_count, 0).astype(np.int32),
        }
        if np.any(done):
            self.reset(np.nonzero(done)[0])
        return self._obs(), rewards, done.astype(np.bool_), info

    def _obs(self) -> np.ndarray:
        extra = np.zeros((self.num_envs, 2), dtype=np.float32)
        extra[:, 0] = self.last_reward
        extra[:, 1] = self.step_count / max(1, self.cfg.episode_length)
        return np.concatenate([self._last_obs, extra], axis=1)

    def close(self) -> None:
        for env in self.envs:
            env.close()


class GymSingleDiscreteVecEnv(BasePublicVecEnv):
    def __init__(self, cfg: PublicEnvConfig):
        self._init_common(cfg)
        self._init_melee_light_sampling()
        self.envs = [self._make_env() for _ in range(self.num_envs)]
        first = self._reset_one(0)
        self._last_obs = np.zeros((self.num_envs, _flatten_obs(first).shape[0]), dtype=np.float32)
        self._last_obs[0] = _flatten_obs(first)
        for i in range(1, self.num_envs):
            self._last_obs[i] = _flatten_obs(self._reset_one(i))
        space = self.envs[0].action_space
        if callable(space):
            space = space("p1")
        if space.__class__.__name__ == "Discrete":
            self._action_mode = "discrete"
            self.action_space_n = int(space.n)
        elif space.__class__.__name__ == "MultiBinary":
            self._action_mode = "multibinary"
            self._binary_dim = int(np.prod(space.shape))
            self.action_space_n = 2 ** self._binary_dim
        else:
            raise ValueError(f"unsupported gym action space: {space}")
        self._init_action_stats(self.action_space_n)
        self.obs_dim = self._last_obs.shape[1] + 2
        self.reset()

    def _init_melee_light_sampling(self) -> None:
        kwargs = self.cfg.env_kwargs
        self._melee_light_agent_character_pool = _parse_int_pool(
            kwargs.get("agent_character_pool"),
            (int(kwargs.get("agent_character", 2)),),
        )
        self._melee_light_opponent_character_pool = _parse_int_pool(
            kwargs.get("opponent_character_pool"),
            (int(kwargs.get("opponent_character", 0)),),
        )
        self.melee_light_agent_character = np.zeros(self.num_envs, dtype=np.int64)
        self.melee_light_opponent_character = np.zeros(self.num_envs, dtype=np.int64)

    def _make_env(self):
        if self.cfg.name == "SlimeVolley-v0":
            import gym
            import slimevolleygym  # noqa: F401

            return gym.make(self.cfg.name, **self.cfg.env_kwargs)
        if self.cfg.name == "melee_light_knockback":
            from latent_policy.melee_light_env import MeleeLightKnockbackEnv

            env_kwargs = dict(self.cfg.env_kwargs)
            env_kwargs.pop("agent_character_pool", None)
            env_kwargs.pop("opponent_character_pool", None)
            frame_skip = int(env_kwargs.get("frame_skip", 4))
            env_kwargs.setdefault("max_episode_frames", self.cfg.episode_length * max(1, frame_skip))
            return MeleeLightKnockbackEnv(**env_kwargs)
        if self.cfg.name == "footsies":
            import footsiesgym

            return footsiesgym.make({"render_mode": None, **self.cfg.env_kwargs})
        import gymnasium as gymnasium

        return gymnasium.make(self.cfg.name, **self.cfg.env_kwargs)

    def _reset_one(self, env_id: int, options: dict[str, Any] | None = None):
        env = self.envs[env_id]
        try:
            try:
                out = env.reset(seed=self.cfg.seed + env_id, options=options)
            except TypeError:
                out = env.reset(seed=self.cfg.seed + env_id) if options is None else env.reset()
        except Exception:
            if self.cfg.name != "melee_light_knockback":
                raise
            self._replace_env(env_id)
            env = self.envs[env_id]
            try:
                out = env.reset(seed=self.cfg.seed + env_id, options=options)
            except TypeError:
                out = env.reset(seed=self.cfg.seed + env_id) if options is None else env.reset()
        obs = out[0] if isinstance(out, tuple) and len(out) == 2 else out
        if self.cfg.name == "footsies" and isinstance(obs, dict):
            return obs["p1"]
        return obs

    def _replace_env(self, env_id: int) -> None:
        old_env = self.envs[env_id]
        try:
            old_env.close()
        except Exception:
            pass
        self.envs[env_id] = self._make_env()

    def _sample_melee_light_characters(self, idx: np.ndarray) -> None:
        if self.cfg.name != "melee_light_knockback" or idx.size == 0:
            return
        agent_pool = self._melee_light_agent_character_pool
        opponent_pool = self._melee_light_opponent_character_pool
        self.melee_light_agent_character[idx] = self.rng.choice(agent_pool, size=idx.size)
        self.melee_light_opponent_character[idx] = self.rng.choice(opponent_pool, size=idx.size)

    def _melee_light_reset_options(self, env_id: int) -> dict[str, Any] | None:
        if self.cfg.name != "melee_light_knockback":
            return None
        return {
            "agent_character": int(self.melee_light_agent_character[env_id]),
            "opponent_character": int(self.melee_light_opponent_character[env_id]),
        }

    def reset(self, indices: np.ndarray | list[int] | None = None) -> np.ndarray:
        idx = np.arange(self.num_envs) if indices is None else np.asarray(indices, dtype=np.int64)
        self.step_count[idx] = 0
        self.episode_return[idx] = 0.0
        self.last_reward[idx] = 0.0
        self.last_agent_action[idx] = -1
        self.last_opp_action[idx] = -1
        self._reset_action_stats(idx)
        self._sample_opponents(idx)
        self._sample_melee_light_characters(idx)
        for env_id in idx:
            env_id_int = int(env_id)
            self._last_obs[env_id_int] = _flatten_obs(
                self._reset_one(env_id_int, options=self._melee_light_reset_options(env_id_int))
            )
            if self.cfg.name == "SlimeVolley-v0":
                self.envs[env_id_int].unwrapped.otherAction = None
        return self._obs()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.int64)
        switched = self._maybe_switch()
        age_at_action = self.steps_since_switch.copy()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=np.bool_)
        opp_actions = np.full(self.num_envs, -1, dtype=np.int64)
        for env_id, env in enumerate(self.envs):
            action = self._map_action(int(actions[env_id]))
            self.last_agent_action[env_id] = int(actions[env_id])
            try:
                if self.cfg.name == "footsies":
                    opp_action = _scripted_action(
                        self._opponent_policy(env_id),
                        self.action_space_n,
                        self.rng,
                        None,
                        int(self.last_agent_action[env_id]),
                        int(self.last_opp_action[env_id]),
                        int(self.step_count[env_id]),
                        None if self.agent_action_counts is None else self.agent_action_counts[env_id],
                        None if self.opp_action_counts is None else self.opp_action_counts[env_id],
                    )
                    self.last_opp_action[env_id] = opp_action
                    opp_actions[env_id] = opp_action
                    out = env.step({"p1": int(action), "p2": int(opp_action)})
                elif self.cfg.name == "SlimeVolley-v0":
                    opp_action = _scripted_action(
                        self._opponent_policy(env_id),
                        self.action_space_n,
                        self.rng,
                        None,
                        int(self.last_agent_action[env_id]),
                        int(self.last_opp_action[env_id]),
                        int(self.step_count[env_id]),
                        None if self.agent_action_counts is None else self.agent_action_counts[env_id],
                        None if self.opp_action_counts is None else self.opp_action_counts[env_id],
                    )
                    self.last_opp_action[env_id] = opp_action
                    opp_actions[env_id] = opp_action
                    env.unwrapped.otherAction = self._map_action(int(opp_action))
                    out = env.step(action)
                elif self.cfg.name == "melee_light_knockback" and getattr(env, "uses_external_opponent", False):
                    opp_action = _melee_light_scripted_action(
                        self._opponent_policy(env_id),
                        self.action_space_n,
                        self.rng,
                        self._last_obs[env_id],
                        int(actions[env_id]),
                        int(self.last_agent_action[env_id]),
                        int(self.last_opp_action[env_id]),
                        int(self.step_count[env_id]),
                        None if self.agent_action_counts is None else self.agent_action_counts[env_id],
                    )
                    self.last_opp_action[env_id] = opp_action
                    opp_actions[env_id] = opp_action
                    out = env.step(int(action), opponent_action=int(opp_action))
                else:
                    out = env.step(action)
            except Exception:
                if self.cfg.name != "melee_light_knockback":
                    raise
                self._replace_env(env_id)
                rewards[env_id] = 0.0
                done[env_id] = True
                continue
            if len(out) == 5:
                obs, reward, terminated, truncated, _ = out
                if self.cfg.name == "footsies":
                    rewards[env_id] = float(reward.get("p1", 0.0))
                    done[env_id] = bool(terminated.get("p1", False) or truncated.get("p1", False))
                    obs = obs.get("p1", self._last_obs[env_id])
                else:
                    rewards[env_id] = float(reward)
                    done[env_id] = bool(terminated or truncated)
            else:
                obs, reward, done_value, _ = out
                rewards[env_id] = float(reward)
                done[env_id] = bool(done_value)
            if not done[env_id]:
                self._last_obs[env_id] = _flatten_obs(obs)

        self._record_actions(actions, opp_actions)
        self.last_reward = rewards.copy()
        self.step_count += 1
        self.steps_since_switch += 1
        self.episode_return += rewards
        forced_done = self.step_count >= self.cfg.episode_length
        done = done | forced_done
        if self.cfg.name == "melee_light_knockback":
            opponent_names = np.asarray([self._melee_light_opponent_name(i) for i in range(self.num_envs)])
        elif self.cfg.name in {"SlimeVolley-v0", "footsies"}:
            opponent_names = np.asarray([self._opponent_policy(i) for i in range(self.num_envs)])
        else:
            opponent_names = np.asarray(["builtin" for _ in range(self.num_envs)])
        info = {
            "switched": switched,
            "opponent_age": age_at_action,
            "opponent_type": self.opponent_idx.copy(),
            "opponent_name": opponent_names,
            "episode_return": np.where(done, self.episode_return, np.nan).astype(np.float32),
            "episode_length": np.where(done, self.step_count, 0).astype(np.int32),
        }
        if self.cfg.name == "melee_light_knockback":
            info["agent_character"] = self.melee_light_agent_character.copy()
            info["opponent_character"] = self.melee_light_opponent_character.copy()
        if np.any(done):
            self.reset(np.nonzero(done)[0])
        return self._obs(), rewards, done.astype(np.bool_), info

    def _melee_light_opponent_name(self, env_id: int) -> str:
        env = self.envs[env_id]
        if not getattr(env, "uses_external_opponent", False):
            return "cpu"
        character_id = int(self.melee_light_opponent_character[env_id])
        character = MELEE_LIGHT_CHARACTER_NAMES.get(character_id, str(character_id))
        return f"{self._opponent_policy(env_id)}:{character}"

    def _map_action(self, action: int):
        if self._action_mode == "discrete":
            return action % self.action_space_n
        bits = [(action >> i) & 1 for i in range(self._binary_dim)]
        return np.asarray(bits, dtype=np.int64)

    def _obs(self) -> np.ndarray:
        extra = np.zeros((self.num_envs, 2), dtype=np.float32)
        extra[:, 0] = self.last_reward
        extra[:, 1] = self.step_count / max(1, self.cfg.episode_length)
        return np.concatenate([self._last_obs, extra], axis=1)

    def close(self) -> None:
        for env in self.envs:
            env.close()
