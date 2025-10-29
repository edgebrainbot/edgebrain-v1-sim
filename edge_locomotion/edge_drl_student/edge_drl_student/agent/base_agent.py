from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all agents (on-policy or off-policy).

    This class defines:
    - shared hyperparameters (learning rate, tau, batch size, etc.)
    - standard helper functions (soft/hard update)
    - abstract interfaces for training and action selection
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        device: str = "cpu",
        buffer_size: int = int(1e5),
        batch_size: int = 256,
        discount: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.005,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.discount = discount
        self.lr = lr
        self.tau = tau
        self.device = torch.device(device)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.iteration = 0
        self.networks: list[nn.Module] = []

    # ==========================================================
    # 🔹 Abstract interfaces (must be implemented in subclass)
    # ==========================================================

    @abstractmethod
    def train_step(self, *args, **kwargs) -> Any:
        """Perform one gradient update from sampled transitions."""
        pass

    @abstractmethod
    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Return an action given current state."""
        pass

    # ==========================================================
    # 🔹 Common utilities
    # ==========================================================

    def register_network(self, net: nn.Module):
        """Register a network to track for parameter updates."""
        self.networks.append(net)
        return net

    def create_optimizer(self, net: nn.Module, lr: float | None = None):
        """Create an AdamW optimizer for given network."""
        return torch.optim.AdamW(net.parameters(), lr or self.lr)

    @staticmethod
    def hard_update(target: nn.Module, source: nn.Module):
        """Copy all weights from source to target."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(sp.data)

    @staticmethod
    def soft_update(target: nn.Module, source: nn.Module, tau: float):
        """Soft-update target network parameters."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)

    def decay_epsilon(self):
        """Decay exploration epsilon (for ε-greedy policies)."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return self.epsilon

    def _prepare_batch(self, batch: Tuple[np.ndarray, ...]) -> Tuple[torch.Tensor, ...]:
        """Convert numpy batch to tensors on correct device."""
        return tuple(torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch)

    def info(self) -> str:
        """Return readable summary of current hyperparameters."""
        info = [f"{k}: {v}" for k, v in self.__dict__.items() if not callable(v) and not k.startswith("_")]
        return "\n".join(info)


