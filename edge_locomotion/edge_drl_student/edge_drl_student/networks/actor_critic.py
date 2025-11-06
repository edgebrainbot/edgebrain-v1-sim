from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from base_network import BaseNetwork
from mlp import MLP

class ActorNetwork(MLP):
    """Actor network using MLP."""

    def __init__(self, 
                 obs_dim: int, 
                 act_dim: int, 
                 hidden_dims: tuple = (256, 256, 256), 
                 activation_fn: type[nn.Module] = nn.ReLU,
                 output_activation: type[nn.Module] = nn.Tanh,
                 action_bound: float = 1.0):
        super().__init__(
            input_dim=obs_dim,
            output_dim=act_dim,
            hidden_dims=hidden_dims,
            activation_fn=nn.ReLU,
            output_activation=nn.Tanh,
            name="actor"
        )
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network."""
        return self.act_limit * super().forward(obs)
    


class ActorCritic(nn.Module):
    """Actor-Critic network using MLP for both actor and critic."""

    def __init__(self, 
                 obs_dim: int, 
                 act_dim: int, 
                 actor_hidden_dims: tuple = (256, 256, 256), 
                 actor_activation_fn: type[nn.Module] = nn.ReLU,
                 critic_hidden_dims: tuple = (256, 256, 256), 
                 critic_activation_fn: type[nn.Module] = nn.ReLU,
                 act_limit: float = 1.0):
        super().__init__()
        self.actor = MLP(obs_dim, act_dim, actor_hidden_dims, actor_activation_fn, act_limit)
        self.critic = MLP(obs_dim, 1, critic_hidden_dims, critic_activation_fn, act_limit)
        self.hidden_dims = hidden_dims
        self.activation_fn = activation_fn
        self.output_activation = output_activation
        super().__init__(input_dim, output_dim, hidden_dims[0], name)


    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from actor network."""
        with torch.no_grad():
            return self.actor(obs)
