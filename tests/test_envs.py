import numpy as np

from latent_policy.envs import SwitchingDuelVecEnv, SwitchingGameConfig


def test_switching_env_shapes_and_done():
    cfg = SwitchingGameConfig(num_envs=4, episode_length=5, switch_hazard=1.0, min_switch_interval=1, seed=7)
    env = SwitchingDuelVecEnv(cfg)
    obs = env.reset()
    assert obs.shape == (4, env.obs_dim)

    saw_done = False
    saw_switch = False
    for _ in range(8):
        obs, rewards, done, info = env.step(np.zeros(4, dtype=np.int64))
        assert obs.shape == (4, env.obs_dim)
        assert rewards.shape == (4,)
        assert done.shape == (4,)
        assert info["opponent_age"].shape == (4,)
        saw_done = saw_done or bool(done.any())
        saw_switch = saw_switch or bool(info["switched"].any())
    assert saw_done
    assert saw_switch
