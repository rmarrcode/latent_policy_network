from latent_policy.ppo import TrainConfig, train


def test_tiny_train_smoke(tmp_path):
    cfg = TrainConfig()
    cfg.run_dir = str(tmp_path)
    cfg.run_name = "smoke"
    cfg.total_updates = 1
    cfg.num_steps = 8
    cfg.env.num_envs = 4
    cfg.env.episode_length = 16
    cfg.policy.agent = "hyper_head"
    cfg.policy.encoder = "mean"
    cfg.policy.context_len = 4
    cfg.policy.hidden_dim = 16
    cfg.policy.latent_dim = 16
    cfg.num_minibatches = 2
    cfg.update_epochs = 1
    cfg.eval_interval = 0
    cfg.save_interval = 1
    cfg.progress = False
    metrics = train(cfg)
    assert "rollout_reward_mean" in metrics
