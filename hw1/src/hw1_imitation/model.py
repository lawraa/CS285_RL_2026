"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        out_dim = chunk_size * action_dim
        layers.append(nn.Linear(in_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # state: (B, state_dim)
        # action_chunk: (B, chunk_size, action_dim)
        B = state.shape[0] # get batch size 
        flat = self.net(state) # (B, chunk_size * action_dim)
        prediction = flat.view(B, self.chunk_size, self.action_dim) # reshape to (B, chunk_size, action_dim)
        return torch.mean((prediction - action_chunk) ** 2) # MSE loss


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # One forward pass to get the action chunk
        B = state.shape[0] # get batch size 
        flat = self.net(state) # shape: (B, chunk_size * action_dim)
        prediction = flat.view(B, self.chunk_size, self.action_dim) # reshape to (B, chunk_size, action_dim)
        return prediction


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.flat_action_dim = chunk_size * action_dim
        in_dim = state_dim + self.flat_action_dim + 1 # +1 for time 
        layers: list[nn.Module] = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            cur = h
        layers.append(nn.Linear(cur, self.flat_action_dim))
        self.v_net = nn.Sequential(*layers)


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        device = state.device
        A_t = action_chunk.view(B, self.flat_action_dim)
        A_t0 = torch.randn(B, self.flat_action_dim, device=device)
        tau = torch.rand(B, 1, device=device)
        x_tau = (1.0 - tau) * A_t0 + tau * A_t
        v_target = A_t - A_t0
        input = torch.cat([state, x_tau, tau], dim=1)
        v_pred =self.v_net(input)
        return torch.mean((v_pred - v_target) ** 2)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        device = state.device
        D = self.flat_action_dim

        x = torch.randn(B, D, device=device)

        n = 1.0 / float(num_steps)
        for k in range(num_steps):
            tau_k = torch.full((B, 1), k * n, device=device)
            inp = torch.cat([state, x, tau_k], dim=1)
            v_pred =self.v_net(inp)
            x = x + n * v_pred
        return x.view(B, self.chunk_size, self.action_dim)
    
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        device = state.device
        D = self.flat_action_dim
        A_t0 = torch.randn(B, D, device=device)

        dt = 1.0 / float(num_steps)
        A_t_tau = A_t0

        for k in range(num_steps):
            tau = torch.full((B, 1), k * dt, device=device)
            input = torch.cat([state, A_t_tau, tau], dim=1)
            v_pred = self.v_net(input)
            A_t_tau = A_t_tau + dt * v_pred
        return A_t_tau.view(B, self.chunk_size, self.action_dim)



PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
