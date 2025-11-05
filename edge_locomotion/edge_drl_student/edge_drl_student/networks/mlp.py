from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from base_network import BaseNetwork

class MLPNetwork(BaseNetwork):
    """Standard Multi-Layer Perceptron (MLP) network."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: tuple[int] | list[int]  =(256, 256, 256),
                 activation_fn: type[nn.Module] =nn.ReLU,
                 output_activation: type[nn.Module] | None =None,
                 name: str ="mlp"):
        """ Initialize MLP

        Args:
            input_dim (int): _description_
            output_dim (int): _description_
            hidden_dims (tuple[int] | list[int], optional): _description_. Defaults to (256, 256, 256).
            activation_fn (type[nn.Module], optional): _description_. Defaults to nn.ReLU.
            output_activation (type[nn.Module] | None, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to "mlp".
        """
        self.hidden_dims = hidden_dims
        self.activation_fn = activation_fn
        self.output_activation = output_activation
        super().__init__(input_dim, output_dim, hidden_dims[0], name)

    def _init_layers(self):
        """Initialize MLP layers dynamically based on hidden_dims."""
        layers = []
        prev_dim = self.input_dim

        # Hidden layers
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(self.activation_fn())
            prev_dim = h_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))

        if self.output_activation is not None:
            layers.append(self.output_activation())

        self.net = nn.Sequential(*layers)
        self.apply(self.init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.net(x)
