from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseNetwork(nn.Module, ABC):
    """Abstract neural network structure used by all agents."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, name: str = "network"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.name = name
        self.iteration = 0
        self._init_layers()

    @abstractmethod
    def _init_layers(self):
        """Define layers in subclass."""
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass logic."""
        pass

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)