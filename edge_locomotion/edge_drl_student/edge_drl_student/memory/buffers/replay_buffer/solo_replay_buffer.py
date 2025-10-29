import torch
from typing import Dict, Generator, Tuple, Optional
from edge_drl_student.memory.buffers.base_buffer import BaseBuffer
from edge_drl_student.memory.samplers.uniform_sampler import UniformSampler
from edge_drl_student.memory.buffers import Transition, TransitionBatch


class SoloReplayBuffer(BaseBuffer):
    """
    SoloReplayBuffer
    ----------------
    A united replay buffer implementation built on top of the base class.
    """

    def __init__(self, 
                 buffer_size: int, 
                 sampler: Optional[UniformSampler] = None, 
                 obs_dim: int = 12,
                 action_dim: int = 6,
                 device: Optional[str] = 'cpu'):
        if sampler is None:
            sampler = UniformSampler(buffer_size)
            
        super().__init__(buffer_size, sampler, device=device)

        self.obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=self.device)

    def add_transition(self, t: Transition):
        """
        Add one transition into the buffer.
        Typically called at every environment step.
        """
        idx = self.idx

        self.obs[idx] = t.obs
        self.actions[idx] = t.action
        self.rewards[idx] = t.reward
        self.next_obs[idx] = t.next_obs
        self.dones[idx] = t.done

        self.idx = (self.idx + 1) % self.buffer_size
        if self.idx == 0:
            self.full = True

    def _get_samples(self, indices: torch.Tensor) -> TransitionBatch:
        """
        Given sampled indices, return the corresponding transitions.
        """
        return TransitionBatch(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            dones=self.dones[indices]
        )
    
    def sample(self, batch_size: int) -> TransitionBatch:
        """
        Sample a batch of transitions using the attached sampler.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            A batch of transitions obtained via `_get_samples()`.
        """
        valid_size = self.size()
        if valid_size <= 0:
            raise ValueError("Cannot sample from an empty buffer.")
        if valid_size > self.buffer_size:
            raise ValueError(
                f"valid_size ({valid_size}) exceeds buffer_size ({self.buffer_size})."
            )

        # Delegate sampling to sampler
        sample_indices = self.sampler.fetch(batch_size, valid_size)
        return self._get_samples(sample_indices)