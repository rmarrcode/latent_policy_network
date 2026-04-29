import pytest
import torch

from latent_policy.models import PolicyConfig, build_policy


@pytest.mark.parametrize("agent", ["static_mlp", "hyper_head", "film", "full_hyper"])
def test_policy_shapes(agent):
    cfg = PolicyConfig(obs_dim=12, action_dim=5, context_len=6, agent=agent, encoder="gru", hidden_dim=32, latent_dim=32)
    policy = build_policy(cfg)
    obs = torch.zeros(3, cfg.obs_dim)
    context = torch.zeros(3, cfg.context_len, cfg.obs_dim)
    action, logprob, entropy, value = policy.get_action_and_value(obs, context)
    assert action.shape == (3,)
    assert logprob.shape == (3,)
    assert entropy.shape == (3,)
    assert value.shape == (3,)
