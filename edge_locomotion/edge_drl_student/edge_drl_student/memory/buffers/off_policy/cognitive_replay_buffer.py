import torch
import numpy as np
from dataclasses import asdict
from typing import Callable, Dict, Generator, Tuple, Optional

from edge_drl_student.memory.buffers import SoloReplayBuffer
from edge_drl_student.memory.buffers.base_buffer import BaseBuffer
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer, ENVELOPE_P
from edge_drl_student.memory.samplers.cognitive_sampler import ACTRCognitiveSampler
from edge_drl_student.memory.samplers.uniform_sampler import UniformSampler
from edge_drl_student.memory.buffers import TransitionData, TransitionBatch
from edge_drl_student.utils.utils import ActionMode, safety_value


class CognitiveReplayBuffer(DualReplayBuffer):
    def __init__(self, 
                 student_buf_size: int, 
                 student_prefill_size: int,
                 teacher_buf_size: int, 
                 teacher_prefill_size: int,
                 obs_dim: int = 12,
                 action_dim: int = 6,
                 device: str = "cpu",
                 ) -> None:

        # if cognitive_sampler is None:
        #     raise ValueError("Cognitive sampler must be provided for CognitiveReplayBuffer.")

        super().__init__(
            student_buf_size=student_buf_size,
            student_prefill_size=student_prefill_size,
            teacher_buf_size=teacher_buf_size,
            teacher_prefill_size=teacher_prefill_size,
            sampler=None,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device
        )
        
        # Initialize cognitive samplers for both buffers
        self.stu_cognitive_sampler = ACTRCognitiveSampler(
            replay_buf=self.stu_buf,
            device=device
        )
        self.tea_cognitive_sampler = ACTRCognitiveSampler(
            replay_buf=self.tea_buf,
            device=device
        )
        
        self.stu_blending_weight = 0.5
        self.tea_blending_weight = 0.5
        
        print("Initialized CognitiveReplayBuffer with cognitive samplers for both student and teacher buffers.")

    def sample(self, batch_size: int) -> TransitionBatch:
        """Override sample method to use cognitive sampling strategy.

        Args:
            batch_size (int): The size of the batch to sample.
        Returns:
            A batch of sampled transitions.
        """
        
        # Invalid batch size check
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        elif batch_size > self.stu_buf.size + self.tea_buf.size:
            raise ValueError(
                f"Requested batch size {L} exceeds total stored transitions "
                f"({self.stu_buf.size + self.tea_buf.size})."
            )

        L = batch_size
        if self.last_data_type == ActionMode.STUDENT:
            stu_idx = self.stu_buf.pointer - 1
            boundary_state = self.stu_buf.obs[stu_idx]

        elif self.last_data_type == ActionMode.TEACHER:
            tea_idx = self.tea_buf.pointer - 1
            boundary_state = self.tea_buf.obs[tea_idx]
        else:
            raise RuntimeError(f"Unknown action type pointer: {self.last_data_type}")

        Vs = safety_value(boundary_state[2:].cpu().numpy(), ENVELOPE_P) * 0.01

        # Batch size for PHY-Teacher Buffer
        tea_batch_size = max(min(L - 1, int(L * Vs)), 1)

        # Batch size for DRL-Student Buffer
        stu_batch_size = L - tea_batch_size
        
        stu_batch = self.stu_cognitive_sampler.fetch(stu_batch_size, self.stu_buf.size) if stu_batch_size > 0 else None
        tea_batch = self.tea_cognitive_sampler.fetch(tea_batch_size, self.tea_buf.size) if tea_batch_size > 0 else None

        print(f"stu_batch_size: {stu_batch_size}, tea_batch_size: {tea_batch_size}")
        if tea_batch_size == 1:
            print(f"tea_batch: {tea_batch}")
        # print(f"stu_batch: {stu_batch}, tea_batch: {tea_batch}")


        stu_dict = asdict(stu_batch) if stu_batch is not None else None
        tea_dict = asdict(tea_batch) if tea_batch is not None else None

        # Merge two batches
        merged_dict = {
            k: torch.cat(
                [b[k] for b in [stu_dict, tea_dict] if b is not None and b[k] is not None],
                dim=0
            )
            for k in (stu_dict.keys() if stu_dict is not None else tea_dict.keys())
        }

        return TransitionBatch(**merged_dict), 

    def get_weighted_blended_action(self) -> Optional[torch.Tensor]:
        """Get the weighted blended action from both cognitive samplers.

        Returns:
            torch.Tensor: The last blended action.
        """
        stu_action = self.stu_cognitive_sampler.get_blending_action()
        tea_action = self.tea_cognitive_sampler.get_blending_action()
        
        if stu_action is None and tea_action is None:
            return None
        elif stu_action is None:
            return tea_action
        elif tea_action is None:
            return stu_action
        else:
            blended_action = (self.stu_blending_weight * stu_action +
                              self.tea_blending_weight * tea_action)
            return blended_action