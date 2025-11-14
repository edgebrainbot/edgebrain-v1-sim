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
                 prefill_size: int,
                 sampler: Optional[BaseSampler] = None, 
                 device: Optional[str] = 'cpu'):
        """
        Args:
            buffer_size (int): Maximum number of transitions the buffer can store.
            prefill_size (int): Number of transitions to prefill before sampling.
            sampler (Optional[BaseSampler]): A sampler used to fetch indices (default: UniformSampler).
            device (Optional[str]): The device to store the buffer on (default: 'cpu').
        """
        
        assert buffer_size > 0, "Buffer size must be positive."
        assert prefill_size >= 0, "Prefill size must be non-negative."
        assert prefill_size <= buffer_size, "Prefill size cannot exceed buffer size."    
        
        self._idx = 0
        self._full = False
        self._history_count = 0
        self.buffer_size = buffer_size
        self.prefill_size = prefill_size
        self.device = device
        
        self.sampler: BaseSampler = sampler or UniformSampler(buffer_size)
        if hasattr(self.sampler, "set_buffer_size"):
            self.sampler.set_buffer_size(buffer_size)

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
        
    def extend(self, t_batch: TransitionBatch):
        """Add multiple transitions into the buffer."""
        raise NotImplementedError("extend() not implemented in BaseBuffer.")

    def set_sampler(self, sampler: BaseSampler):
        """Replace the sampler dynamically (e.g., switch to prioritized)."""
        self.sampler = sampler
        self.sampler.set_buffer_size(self.buffer_size)

    @property
    def pointer(self) -> int:
        """Current circular write pointer."""
        return self._idx

    @property
    def size(self) -> int:
        """Return number of valid transitions currently stored."""
        return self.buffer_size if self._full else self._idx

    @property
    def capacity(self) -> int:
        """Return maximum buffer capacity."""
        return self.buffer_size

    @property
    def is_ready(self) -> bool:
        """
        Whether buffer has enough transitions for sampling.
        Returns True when size >= prefill_size.
        """
        return self.size >= self.prefill_size
    
    @property
    def is_prefilled(self) -> bool:
        """Deprecated: use is_ready instead."""
        import warnings
        warnings.warn(
            "is_prefilled is deprecated, use is_ready",
            DeprecationWarning,
            stacklevel=2
        )
        return self.is_ready
    
    @property
    def history_count(self) -> int:
        """Total number of transitions ever written into the buffer (including overwrites)."""
        return self._history_count

    def reset(self):
        """Reset write pointer but DO NOT clear existing data."""
        self._idx = 0
        self._full = False
        self._history_count = 0
        
    def clear(self):
        """
        Clear stored tensor data for replay buffers that define them.
        """
        for attr in ("obs", "actions", "rewards", "next_obs", "dones"):
            if hasattr(self, attr):
                getattr(self, attr).zero_()
        self.reset()
        

    def __len__(self) -> int:
        return self.size
