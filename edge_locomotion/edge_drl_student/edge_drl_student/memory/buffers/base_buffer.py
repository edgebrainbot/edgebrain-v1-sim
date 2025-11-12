from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, List
from dataclasses import dataclass
import torch

from edge_drl_student.memory.samplers import BaseSampler
from edge_drl_student.memory.samplers import UniformSampler

@dataclass
class TransitionData:
    """Batch-level semantic wrapper for sampled experiences."""
    obs: torch.Tensor       # Observation
    action: torch.Tensor    # Action taken
    reward: torch.Tensor    # Reward received
    next_obs: torch.Tensor  # Next observation
    done: torch.Tensor      # Whether the episode is done
    info: Any = None        # Additional info

@dataclass
class TransitionBatch:
    """Batched transitions sampled from the replay buffer."""
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    info: Optional[List[Any]] = None

    def to(self, device: str) -> "TransitionBatch":
        """Move all tensors to target device."""
        return TransitionBatch(
            obs=self.obs.to(device),
            action=self.action.to(device),
            reward=self.reward.to(device),
            next_obs=self.next_obs.to(device),
            done=self.done.to(device),
            info=self.info
        )

    def __len__(self):
        """Batch size."""
        return self.obs.shape[0]
    

class BaseBuffer(ABC):
    
    """
    BaseBuffer (abstract class)
    ---------------------------
    A lightweight and extendable replay buffer interface for single-environment use.

    Handles storage management (idx, full, buffer_size) and delegates
    sampling logic to a pluggable Sampler.
    """

    def __init__(self, 
                 buffer_size: int, 
                 sampler: Optional[BaseSampler] = None, 
                 device: Optional[str] = 'cpu'):
        """
        Args:
            buffer_size (int): Maximum number of transitions the buffer can store.
            sampler (Optional[BaseSampler]): A sampler used to fetch indices (default: UniformSampler).
            device (Optional[str]): The device to store the buffer on (default: 'cpu').
        """
        self.buffer_size = buffer_size
        self.idx = 0
        self.full = False
        self.sampler: BaseSampler = sampler or UniformSampler(buffer_size)
        self.device = device

    @abstractmethod
    def add_transition(self, *args, **kwargs):
        """
        Add one transition into the buffer.
        Typically called at every environment step.

        Subclasses may extend arguments, e.g.:
        - SoloReplayBuffer.add_transition(transition)
        - DualReplayBuffer.add_transition(transition, action_mode)
        """
        raise NotImplementedError

    def _get_samples(self, indices: torch.Tensor):
        """
        Given sampled indices, return the corresponding transitions.
        Should return a structured batch (e.g., dict, dataclass, namedtuple).
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int):
        """
        Sample a batch of transitions using the attached sampler.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            A batch of transitions obtained via `_get_samples()`.
        """
        raise NotImplementedError

    def to_transition(self, obs, action, reward, next_obs, done, info=None) -> TransitionData:
        """Convert the tuples to Transition version."""

        def to_torch(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=self.device, dtype=torch.float32)
            return torch.as_tensor(x, device=self.device, dtype=torch.float32)

        return TransitionData(
            obs=to_torch(obs),
            action=to_torch(action),
            reward=to_torch(reward),
            next_obs=to_torch(next_obs),
            done=to_torch(done),
            info=to_torch(info) if info is not None else None,
        )

    def set_sampler(self, sampler: BaseSampler):
        """Replace the sampler dynamically (e.g., switch to prioritized)."""
        self.sampler = sampler
        self.sampler.set_buffer_size(self.buffer_size)

    @property
    def size(self) -> int:
        """Return number of valid transitions currently stored."""
        return self.buffer_size if self.full else self.idx

    def capacity(self) -> int:
        """Return maximum buffer capacity."""
        return self.buffer_size

    def reset(self):
        """Reset write pointer."""
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.size()
