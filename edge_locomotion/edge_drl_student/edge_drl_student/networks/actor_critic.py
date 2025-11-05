from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from base_network import BaseNetwork
from mlp import MLPNetwork

class ActorCritic(nn.Module):
    """Actor-Critic network using MLP for both actor and critic."""

    def __init__(self, 
                 obs_dim: int, 
                 act_dim: int, 
                 hidden_dims: tuple = (256, 256), 
                 act_limit: float = 1.0):
        super().__init__()
        self.actor = MLPNetwork(obs_dim, act_dim, hidden_dims, act_limit)
        self.critic = MLPNetwork(obs_dim, act_dim, hidden_dims, act_limit)
        self.hidden_dims = hidden_dims
        self.activation_fn = activation_fn
        self.output_activation = output_activation
        super().__init__(input_dim, output_dim, hidden_dims[0], name)


    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from actor network."""
        with torch.no_grad():
            return self.actor(obs)
