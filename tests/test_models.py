import pytest
import torch

from latent_policy.models import PolicyConfig, build_policy


@pytest.mark.parametrize("agent", ["static_mlp", "context_mlp", "hyper_head", "film", "full_hyper"])
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


def test_context_mlp_uses_history_with_fixed_actor_weights():
    cfg = PolicyConfig(obs_dim=12, action_dim=5, agent="context_mlp", encoder="gru",
                       hidden_dim=32, encoder_hidden_dim=16, latent_dim=16, context_len=6)
    policy = build_policy(cfg)
    obs = torch.zeros(3, 12)
    first_logits, _ = policy.logits_and_value(obs, torch.zeros(3, 6, 12))
    second_logits, _ = policy.logits_and_value(obs, torch.ones(3, 6, 12))
    assert not torch.allclose(first_logits, second_logits)
    second_logits.sum().backward()
    assert policy.encoder.gru.weight_ih_l0.grad.abs().sum() > 0
    assert isinstance(policy.actor, torch.nn.Linear)
    assert not any("gen" in name or "modulator" in name for name, _ in policy.named_parameters())


def test_policy_action_path_sanitizes_nonfinite_outputs():
    cfg = PolicyConfig(obs_dim=12, action_dim=5, context_len=6, agent="full_hyper", encoder="gru", hidden_dim=32, latent_dim=32)
    policy = build_policy(cfg)
    with torch.no_grad():
        for param in policy.parameters():
            param.fill_(float("nan"))
            break

    obs = torch.zeros(3, cfg.obs_dim)
    context = torch.zeros(3, cfg.context_len, cfg.obs_dim)
    action, logprob, entropy, value = policy.get_action_and_value(obs, context)

    assert action.shape == (3,)
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(value).all()
