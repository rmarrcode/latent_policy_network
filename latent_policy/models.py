from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F


@dataclass
class PolicyConfig:
    obs_dim: int
    action_dim: int
    context_len: int = 16
    agent: str = "hyper_head"
    encoder: str = "gru"
    hidden_dim: int = 64
    latent_dim: int = 64
    generated_hidden_dim: int = 24
    attention_heads: int = 4
    attention_layers: int = 1
    weight_scale: float = 0.8


def mlp(sizes: Iterable[int], activation: type[nn.Module] = nn.Tanh, last_std: float | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    sizes = list(sizes)
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.orthogonal_(linear.weight, gain=1.0 if i == len(sizes) - 2 else 1.41)
        nn.init.zeros_(linear.bias)
        if last_std is not None and i == len(sizes) - 2:
            nn.init.normal_(linear.weight, std=last_std)
        layers.append(linear)
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def init_last_linear(module: nn.Module, std: float = 0.01) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, std=std)
            nn.init.zeros_(layer.bias)
            return


class ContextEncoder(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        encoder = cfg.encoder
        if encoder == "flat":
            self.kind = "flat"
            self.net = mlp([cfg.context_len * cfg.obs_dim, cfg.hidden_dim, cfg.latent_dim])
        elif encoder == "mean":
            self.kind = "mean"
            self.net = mlp([cfg.obs_dim, cfg.hidden_dim, cfg.latent_dim])
        elif encoder == "gru":
            self.kind = "gru"
            self.gru = nn.GRU(cfg.obs_dim, cfg.hidden_dim, batch_first=True)
            self.net = mlp([cfg.hidden_dim, cfg.latent_dim])
        elif encoder == "attention":
            self.kind = "attention"
            self.input = nn.Linear(cfg.obs_dim, cfg.hidden_dim)
            self.pos = nn.Parameter(torch.zeros(1, cfg.context_len, cfg.hidden_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=cfg.attention_heads,
                dim_feedforward=cfg.hidden_dim * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.attention_layers)
            self.net = mlp([cfg.hidden_dim, cfg.latent_dim])
        else:
            raise ValueError(f"unknown context encoder: {encoder}")

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if self.kind == "flat":
            return self.net(context.flatten(1))
        if self.kind == "mean":
            return self.net(context.mean(dim=1))
        if self.kind == "gru":
            _, hidden = self.gru(context)
            return self.net(hidden[-1])
        if self.kind == "attention":
            x = self.input(context) + self.pos[:, : context.shape[1]]
            x = self.transformer(x)
            return self.net(x[:, -1])
        raise RuntimeError("unreachable")


class ActorCritic(nn.Module):
    cfg: PolicyConfig

    def logits_and_value(self, obs: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _stable_logits_and_value(self, obs: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value = self.logits_and_value(obs, context)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self._stable_logits_and_value(obs, context)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, context: torch.Tensor, deterministic: bool) -> torch.Tensor:
        logits, _ = self._stable_logits_and_value(obs, context)
        if deterministic:
            return logits.argmax(dim=-1)
        return Categorical(logits=logits).sample()


class StaticMLPPolicy(ActorCritic):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = mlp([cfg.obs_dim, cfg.hidden_dim, cfg.hidden_dim])
        self.actor = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.critic = nn.Linear(cfg.hidden_dim, 1)
        nn.init.normal_(self.actor.weight, std=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.normal_(self.critic.weight, std=1.0)
        nn.init.zeros_(self.critic.bias)

    def logits_and_value(self, obs: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del context
        h = self.trunk(obs)
        return self.actor(h), self.critic(h)


class HyperHeadPolicy(ActorCritic):
    """Generates the final actor head from context and uses a learned critic."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ContextEncoder(cfg)
        self.obs_trunk = mlp([cfg.obs_dim, cfg.hidden_dim, cfg.hidden_dim])
        self.head_gen = mlp(
            [cfg.latent_dim, cfg.hidden_dim, cfg.action_dim * cfg.hidden_dim + cfg.action_dim]
        )
        self.critic = mlp([cfg.hidden_dim + cfg.latent_dim, cfg.hidden_dim, 1])
        init_last_linear(self.head_gen, std=0.01)
        init_last_linear(self.critic, std=1.0)

    def logits_and_value(self, obs: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(context)
        h = self.obs_trunk(obs)
        params = torch.tanh(self.head_gen(z)) * self.cfg.weight_scale
        weight_len = self.cfg.action_dim * self.cfg.hidden_dim
        weights = params[:, :weight_len].view(-1, self.cfg.action_dim, self.cfg.hidden_dim)
        bias = params[:, weight_len:].view(-1, self.cfg.action_dim)
        logits = torch.bmm(weights, h.unsqueeze(-1)).squeeze(-1) + bias
        value = self.critic(torch.cat([h, z], dim=-1))
        return logits, value


class FiLMPolicy(ActorCritic):
    """Uses context to modulate policy features with generated FiLM parameters."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ContextEncoder(cfg)
        self.fc1 = nn.Linear(cfg.obs_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.modulator = mlp([cfg.latent_dim, cfg.hidden_dim, 4 * cfg.hidden_dim])
        self.actor = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.critic = mlp([cfg.hidden_dim + cfg.latent_dim, cfg.hidden_dim, 1])
        init_last_linear(self.modulator, std=0.01)
        nn.init.normal_(self.actor.weight, std=0.01)
        nn.init.zeros_(self.actor.bias)
        init_last_linear(self.critic, std=1.0)

    def logits_and_value(self, obs: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(context)
        gamma1, beta1, gamma2, beta2 = self.modulator(z).chunk(4, dim=-1)
        h = self.fc1(obs)
        h = F.relu(h * (1.0 + torch.tanh(gamma1)) + beta1)
        h = self.fc2(h)
        h = F.relu(h * (1.0 + torch.tanh(gamma2)) + beta2)
        return self.actor(h), self.critic(torch.cat([h, z], dim=-1))


class FullHyperPolicy(ActorCritic):
    """Generates all weights for a compact one-hidden-layer actor network."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ContextEncoder(cfg)
        gh = cfg.generated_hidden_dim
        self.param_len = cfg.obs_dim * gh + gh + gh * cfg.action_dim + cfg.action_dim
        self.actor_gen = mlp([cfg.latent_dim, cfg.hidden_dim, self.param_len])
        self.critic = mlp([cfg.obs_dim + cfg.latent_dim, cfg.hidden_dim, cfg.hidden_dim, 1])
        init_last_linear(self.actor_gen, std=0.01)
        init_last_linear(self.critic, std=1.0)

    def logits_and_value(self, obs: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(context)
        params = torch.tanh(self.actor_gen(z)) * self.cfg.weight_scale
        bsz = obs.shape[0]
        gh = self.cfg.generated_hidden_dim
        cursor = 0
        w1_len = self.cfg.obs_dim * gh
        w1 = params[:, cursor : cursor + w1_len].view(bsz, gh, self.cfg.obs_dim)
        cursor += w1_len
        b1 = params[:, cursor : cursor + gh].view(bsz, gh)
        cursor += gh
        w2_len = gh * self.cfg.action_dim
        w2 = params[:, cursor : cursor + w2_len].view(bsz, self.cfg.action_dim, gh)
        cursor += w2_len
        b2 = params[:, cursor : cursor + self.cfg.action_dim].view(bsz, self.cfg.action_dim)

        h = torch.bmm(w1, obs.unsqueeze(-1)).squeeze(-1) + b1
        h = torch.tanh(h)
        logits = torch.bmm(w2, h.unsqueeze(-1)).squeeze(-1) + b2
        value = self.critic(torch.cat([obs, z], dim=-1))
        return logits, value


def build_policy(cfg: PolicyConfig) -> ActorCritic:
    if cfg.agent == "static_mlp":
        return StaticMLPPolicy(cfg)
    if cfg.agent == "hyper_head":
        return HyperHeadPolicy(cfg)
    if cfg.agent == "film":
        return FiLMPolicy(cfg)
    if cfg.agent == "full_hyper":
        return FullHyperPolicy(cfg)
    raise ValueError(f"unknown agent: {cfg.agent}")


def count_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)
