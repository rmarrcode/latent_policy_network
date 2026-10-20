import json
from dataclasses import replace

import pytest

from latent_policy.compare import compare, match_baseline, parameter_count
from latent_policy.models import PolicyConfig, build_policy, count_parameters
from latent_policy.ppo import TrainConfig


@pytest.mark.parametrize("agent", ["hyper_head", "full_hyper", "film"])
@pytest.mark.parametrize("encoder", ["mean", "flat", "gru", "attention"])
def test_match_baseline_uses_closest_width_and_real_parameter_counts(agent, encoder):
    latent = PolicyConfig(obs_dim=12, action_dim=5, agent=agent, encoder=encoder,
                          hidden_dim=32, latent_dim=32, context_len=6)
    baseline = match_baseline(latent)
    target = count_parameters(build_policy(latent))
    actual = count_parameters(build_policy(baseline))
    assert parameter_count(latent) == target
    assert parameter_count(baseline) == actual
    assert abs(actual - target) / target <= 0.01
    for width in (max(1, baseline.hidden_dim - 1), baseline.hidden_dim + 1):
        assert abs(actual - target) <= abs(parameter_count(replace(baseline, hidden_dim=width)) - target)
    assert baseline.agent == "static_mlp"
    assert baseline.encoder == "mean"
    assert latent.hidden_dim == 32


def test_match_baseline_rejects_unmatched_pair():
    latent = PolicyConfig(obs_dim=12, action_dim=5)
    with pytest.raises(ValueError, match="exceeds"):
        match_baseline(latent, tolerance=0)


@pytest.mark.parametrize("tolerance", [-0.1, 1.1, float("nan")])
def test_match_baseline_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        match_baseline(PolicyConfig(obs_dim=12, action_dim=5), tolerance)


def test_comparison_trains_matched_pair_and_evaluates(tmp_path):
    cfg = TrainConfig()
    cfg.total_updates = 1
    cfg.num_steps = 4
    cfg.env.num_envs = 2
    cfg.env.episode_length = 4
    cfg.policy.hidden_dim = 32
    cfg.policy.latent_dim = 32
    cfg.policy.context_len = 4
    cfg.num_minibatches = 2
    cfg.update_epochs = 1
    cfg.eval_interval = 0
    cfg.eval_episodes = 2
    cfg.save_interval = 0
    cfg.progress = False
    cfg.device = "cpu"
    rows = compare(cfg, [7], tmp_path)
    assert [row["agent"] for row in rows] == ["static_mlp", "hyper_head"]
    assert rows[0]["global_step"] == rows[1]["global_step"] == 8
    assert all(row["seed"] == 7 and row["eval_episodes"] == 2 for row in rows)
    assert abs(rows[0]["params"] - rows[1]["params"]) / rows[1]["params"] <= 0.01
    plan = json.loads((tmp_path / "comparison_plan.json").read_text())
    assert plan["baseline_params"] == rows[0]["params"]
    assert plan["latent_params"] == rows[1]["params"]
    assert plan["training_config"]["eval_interval"] == 1
    assert cfg.eval_interval == 0
    assert (tmp_path / "summary.csv").exists()


def test_comparison_dry_run_does_not_train(tmp_path):
    assert compare(TrainConfig(), [1], tmp_path, dry_run=True) == []
    assert (tmp_path / "comparison_plan.json").exists()
    assert not (tmp_path / "summary.csv").exists()
