import numpy as np
import pytest

from latent_policy.public_envs import PublicEnvConfig, _scripted_action, build_public_env


@pytest.mark.parametrize(
    "cfg",
    [
        PublicEnvConfig(kind="openspiel_matrix", name="matrix_rps", num_envs=2, episode_length=4),
        PublicEnvConfig(kind="openspiel_turn", name="kuhn_poker", num_envs=2, episode_length=4),
        PublicEnvConfig(
            kind="pettingzoo_parallel",
            name="pettingzoo.classic.rps_v2:parallel_env",
            train_agent="player_0",
            num_envs=2,
            episode_length=4,
            max_cycles=4,
            opponent_pool=("majority_agent", "counter_majority"),
            env_kwargs={"num_actions": 3},
        ),
    ],
)
def test_public_vec_env_step(cfg):
    env = build_public_env(cfg)
    obs = env.reset()
    assert obs.shape == (cfg.num_envs, env.obs_dim)
    actions = np.zeros(cfg.num_envs, dtype=np.int64)
    for _ in range(2):
        obs, rewards, done, info = env.step(actions)
        assert obs.shape == (cfg.num_envs, env.obs_dim)
        assert rewards.shape == (cfg.num_envs,)
        assert done.shape == (cfg.num_envs,)
        assert info["episode_return"].shape == (cfg.num_envs,)
    env.close()


def test_scripted_action_adaptive_policies():
    rng = np.random.default_rng(0)
    counts = np.asarray([0, 5, 1], dtype=np.int32)

    assert _scripted_action("majority_agent", 3, rng, None, -1, -1, 0, agent_action_counts=counts) == 1
    assert _scripted_action("minority_agent", 3, rng, None, -1, -1, 0, agent_action_counts=counts) == 0
    assert _scripted_action("counter_majority", 3, rng, None, -1, -1, 0, agent_action_counts=counts) == 2
