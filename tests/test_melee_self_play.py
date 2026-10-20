import json

import numpy as np
import pytest
import torch

from latent_policy.melee_elo import evaluate_checkpoints, fit_elo
from latent_policy.melee_env import OBS_DIM, MeleeConfig
from latent_policy.melee_self_play import SelfPlayConfig, policy_configs, train_comparison, validate
from latent_policy.ppo import TrainConfig


class FakeDolphinEnv:
    instances = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.elapsed = 0
        self.closed = False
        self.__class__.instances.append(self)

    def reset(self):
        self.elapsed = 0
        return np.zeros((2, OBS_DIM), dtype=np.float32)

    def step(self, actions):
        self.elapsed += 1
        reward = float(np.sign(actions[0] - actions[1])) if self.elapsed == 2 else 0.0
        return np.full((2, OBS_DIM), 0.01, dtype=np.float32), np.array([reward, -reward]), self.elapsed == 2, {"frames": self.elapsed}

    def close(self):
        self.closed = True


def tiny_config():
    cfg = TrainConfig(device="cpu", total_updates=2, num_steps=4, num_minibatches=2,
                      update_epochs=1, target_kl=None, torch_threads=1)
    cfg.policy.agent = "full_hyper"
    cfg.policy.hidden_dim = 32
    cfg.policy.latent_dim = 16
    cfg.policy.generated_hidden_dim = 8
    cfg.policy.context_len = 2
    return cfg


def test_training_and_tournament_pipeline_with_test_backend(tmp_path, monkeypatch):
    from latent_policy import melee_self_play

    observed_pools = []
    original_collect = melee_self_play.collect_rollout

    def record_pool(env, policy, opponents, *args):
        observed_pools.append([sum(parameter.sum().item() for parameter in opponent.parameters()) for opponent in opponents])
        return original_collect(env, policy, opponents, *args)

    monkeypatch.setattr(melee_self_play, "collect_rollout", record_pool)
    cfg = tiny_config()
    options = SelfPlayConfig(seeds=(1,), checkpoint_interval=1, eval_games_per_side=1, eval_warmup_bouts=1)
    manifest = train_comparison(cfg, MeleeConfig(), options, tmp_path, env_factory=FakeDolphinEnv)
    assert len(manifest) == 6
    assert observed_pools[0] == observed_pools[1]
    assert observed_pools[2] == observed_pools[3]
    assert len(observed_pools[2]) == 4
    assert {row["env_steps"] for row in manifest} == {0, 4, 8}
    counts = [row["params"] for row in manifest[:2]]
    assert abs(counts[0] - counts[1]) / counts[1] <= options.param_tolerance
    initial = torch.load(manifest[0]["path"], weights_only=False)
    trained = torch.load(manifest[-2]["path"], weights_only=False)
    assert any(not torch.equal(initial["model_state"][name], trained["model_state"][name]) for name in initial["model_state"])
    assert FakeDolphinEnv.instances[-1].closed
    rows = evaluate_checkpoints(tmp_path, "cpu", env_factory=FakeDolphinEnv)
    assert len(rows) == 6
    assert np.isclose(np.mean([row["elo"] for row in rows]), 1500)
    assert all(np.isfinite(row["elo"]) for row in rows)
    assert FakeDolphinEnv.instances[-1].closed
    for name in ("elo.csv", "matches.csv", "elo_over_time.png", "elo_over_time.svg"):
        assert (tmp_path / name).stat().st_size > 0
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    with pytest.raises(ValueError, match="already contains"):
        train_comparison(cfg, MeleeConfig(), options, tmp_path, env_factory=FakeDolphinEnv)

    class PortBiasedEnv(FakeDolphinEnv):
        def step(self, actions):
            obs, _, done, info = super().step(actions)
            return obs, np.array([1.0, -1.0]) if done else np.zeros(2), done, info

    balanced = evaluate_checkpoints(tmp_path, "cpu", env_factory=PortBiasedEnv)
    np.testing.assert_allclose([row["elo"] for row in balanced], 1500)


def test_elo_fitting_ranks_winner_and_is_independent_of_match_order():
    results = [(0, 1, 8.0, 10), (0, 2, 9.0, 10), (1, 2, 7.0, 10)]
    ratings = fit_elo(3, results)
    assert ratings[0] > ratings[1] > ratings[2]
    assert np.isclose(ratings.mean(), 1500)
    np.testing.assert_allclose(ratings, fit_elo(3, list(reversed(results))))
    np.testing.assert_allclose(fit_elo(2, [(0, 1, 5.0, 10)]), [1500, 1500])


def test_pool_configuration_and_equal_optimization_budget_are_validated():
    cfg = tiny_config()
    cfg.target_kl = 0.04
    with pytest.raises(ValueError, match="target_kl"):
        validate(cfg, SelfPlayConfig())
    cfg.target_kl = None
    with pytest.raises(ValueError, match="positive"):
        validate(cfg, SelfPlayConfig(pool_size_per_agent=0))
    cfg.policy.agent = "static_mlp"
    with pytest.raises(ValueError, match="latent agent"):
        policy_configs(cfg, SelfPlayConfig())
