import json
import csv

import numpy as np
import pytest
import torch

from latent_policy.melee_elo import evaluate_checkpoints, fit_elo, select_pairings
from latent_policy.melee_env import OBS_DIM, MeleeConfig
from latent_policy.melee_self_play import SelfPlayConfig, collect_rollout, policy_configs, train_comparison, validate
from latent_policy.models import PolicyConfig, build_policy, count_parameters
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


def test_balanced_elo_graph_is_connected_and_covers_every_checkpoint():
    pairs = select_pairings(75, 8)
    assert len(pairs) == len(set(pairs)) == 300
    assert all(sum(index in pair for pair in pairs) == 8 for index in range(75))
    reached = {0}
    for _ in range(75):
        reached |= {other for first, second in pairs if first in reached or second in reached for other in (first, second)}
    assert len(reached) == 75
    assert pairs == select_pairings(75, 8)
    assert len(select_pairings(4, 0)) == 6


def test_learner_history_survives_hidden_opponent_switch():
    torch.set_num_threads(1)
    learner_cfg = PolicyConfig(obs_dim=OBS_DIM, action_dim=22, agent="context_mlp", context_len=2,
                               hidden_dim=16, latent_dim=16)
    learner = build_policy(learner_cfg)
    opponents = [build_policy(learner_cfg), build_policy(learner_cfg)]
    cfg = tiny_config()
    cfg.num_steps = 6
    options = SelfPlayConfig(opponent_bouts=1)
    batch, metrics = collect_rollout(FakeDolphinEnv(MeleeConfig()), learner, opponents, cfg, options, 123)
    assert torch.count_nonzero(batch["context"][0]) == 0
    assert torch.count_nonzero(batch["context"][2]) > 0
    assert torch.count_nonzero(batch["context"][4]) > 0
    assert [record["bout_since_switch"] for record in metrics["bout_records"]] == [0, 0, 0]
    assert len({record["port"] for record in metrics["bout_records"]}) == 1


def test_five_arm_multi_seed_comparison_and_switch_evaluation(tmp_path, monkeypatch):
    from latent_policy import melee_self_play

    torch.set_num_threads(1)
    cfg = tiny_config()
    cfg.total_updates = 1
    cfg.policy.hidden_dim = cfg.policy.latent_dim = cfg.policy.encoder_hidden_dim = 64
    cfg.policy.generated_hidden_dim = 24
    options = SelfPlayConfig(agents=("static_mlp", "context_mlp", "film", "hyper_head", "full_hyper"),
                             seeds=(1, 2), checkpoint_interval=1, eval_games_per_side=1,
                             eval_warmup_bouts=1, eval_opponents_per_checkpoint=2, eval_switch_games=2)
    configs = policy_configs(cfg, options)
    counts = [count_parameters(build_policy(policy_cfg)) for policy_cfg in configs]
    assert all(abs(count - counts[-1]) / counts[-1] <= 0.01 for count in counts)
    encoders = []
    for policy_cfg in configs[1:]:
        torch.manual_seed(7)
        policy = build_policy(policy_cfg)
        assert policy_cfg.encoder == "gru"
        assert policy.encoder.gru.hidden_size == 64
        encoders.append(policy.encoder.state_dict())
    for encoder in encoders[1:]:
        assert all(torch.equal(encoder[name], encoders[0][name]) for name in encoder)
    recorded_pools = []
    original = melee_self_play.collect_rollout

    def record_pool(env, policy, opponents, *args):
        recorded_pools.append([(snapshot.cfg.agent, snapshot.self_play_update) for snapshot in opponents])
        return original(env, policy, opponents, *args)

    monkeypatch.setattr(melee_self_play, "collect_rollout", record_pool)
    manifest = train_comparison(cfg, MeleeConfig(), options, tmp_path, env_factory=FakeDolphinEnv)
    assert len(manifest) == 20
    assert all(pool == recorded_pools[0] for pool in recorded_pools[:5])
    assert all(pool == recorded_pools[5] for pool in recorded_pools[5:])
    with (tmp_path / "training.csv").open() as handle:
        metrics = list(csv.DictReader(handle))
    assert len(metrics) == 10
    assert {row["agent"] for row in metrics} == set(options.agents)
    assert all(int(row["env_steps"]) == 4 for row in metrics)
    assert all(float(row["optimization_wall_seconds"]) > 0 for row in metrics)
    assert (tmp_path / "inference_cost.csv").exists()
    assert (tmp_path / "training_bouts.csv").exists()
    evaluate_checkpoints(tmp_path, "cpu", env_factory=FakeDolphinEnv)
    with (tmp_path / "switch_bouts.csv").open() as handle:
        switches = list(csv.DictReader(handle))
    assert len(switches) == 200
    assert all(row["seed"] != row["target_seed"] for row in switches)
    assert (tmp_path / "switch_adaptation.png").stat().st_size > 0
    assert (tmp_path / "evaluation_plan.json").exists()
