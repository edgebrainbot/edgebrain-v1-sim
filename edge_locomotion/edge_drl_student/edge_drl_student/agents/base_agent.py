# base_agent.py
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np


class BaseAgent(ABC):
    """
    Minimal abstract base class for RL agents.
    Defines only the essential interfaces and utilities.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    # Abstract interfaces
    @abstractmethod
    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select action given observation."""
        pass

    @abstractmethod
    def train_step(self, *args, **kwargs):
        """Perform one gradient update."""
        pass

