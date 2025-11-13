import torch
import numpy as np
from dataclasses import asdict
from typing import Callable, Dict, Generator, Tuple, Optional

from edge_drl_student.memory.buffers import SoloReplayBuffer
from edge_drl_student.memory.buffers.base_buffer import BaseBuffer
from edge_drl_student.memory.samplers.actr_declarative_sampler import ACTRDeclarativeSampler
from edge_drl_student.memory.samplers.uniform_sampler import UniformSampler
from edge_drl_student.memory.buffers import TransitionData, TransitionBatch
from edge_drl_student.utils.utils import ActionMode, safety_value

ENVELOPE_P = np.array([[100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0.25, -0, 0, -0, -0, -0, -0, -0, -0],
                       [0, -0, 0.25, 0, -0, -0, -0, -0, -0, -0],
                       [0, 0, 0, 0.6944, -0, 0, 0, 0, 0, 0],
                       [0, -0, -0, -0, 11.1111, -0, -0, -0, -0, -0],
                       [0, -0, -0, 0, -0, 0.25, -0, -0, -0, -0],
                       [0, -0, -0, 0, -0, -0, 0.25, -0, -0, -0],
                       [0, -0, -0, 0, -0, -0, -0, 0.25, -0, -0],
                       [0, -0, -0, 0, -0, -0, -0, -0, 0.25, -0],
                       [0, -0, -0, 0, -0, -0, -0, -0, -0, 4]])

class DeclarativeReplayBuffer(BaseBuffer):
    def __init__(self, 
                 student_buf_size: int, 
                 teacher_buf_size: int, 
                 sampler: Optional[ACTRDeclarativeSampler] = None,
                 obs_dim: int = 12,
                 action_dim: int = 6,
                 device: str = "cpu",
                 ) -> None:

        if sampler is None:
            sampler = UniformSampler(max(student_buf_size, teacher_buf_size))

        self.student_buf_size = student_buf_size       
        self.teacher_buf_size = teacher_buf_size        
        self.device = device

        # Self-learning DRL-Student
        self.stu_buf = SoloReplayBuffer(buffer_size=self.student_buf_size, 
                                        obs_dim=obs_dim, 
                                        action_dim=action_dim,
                                        sampler=sampler,
                                        device=self.device)

        # Teaching-to-learn PHY-Teacher
        self.tea_buf = SoloReplayBuffer(buffer_size=self.teacher_buf_size, 
                                        obs_dim=obs_dim, 
                                        action_dim=action_dim,
                                        sampler=sampler,
                                        device=self.device)

        self.last_data_type = ActionMode.UNCERTAIN


    @property
    def stu_buf_size(self) -> int:
        return self.stu_buf.size

    @property
    def tea_buf_size(self) -> int:
        return self.tea_buf.size

    def add_transition(self, transition: TransitionData, action_mode: ActionMode) -> None:
        """Appends a dataset of transitions to the storage.

        Args:
            dataset (Dataset): The dataset of transitions.
        """

        if action_mode == ActionMode.TEACHER:
            self.tea_buf.add_transition(t=transition)
            self.last_data_type = ActionMode.TEACHER

        elif action_mode == ActionMode.STUDENT:
            self.stu_buf.add_transition(t=transition)
            self.last_data_type = ActionMode.STUDENT

        else:
            raise RuntimeError(f"Unrecognized action mode: {action_mode} for this transition.")


    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample a batch of transitions using the attached sampler.

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
            stu_idx = self.stu_buf.idx - 1
            boundary_state = self.stu_buf.obs[stu_idx]

        elif self.last_data_type == ActionMode.TEACHER:
            tea_idx = self.tea_buf.idx - 1
            boundary_state = self.tea_buf.obs[tea_idx]
        else:
            raise RuntimeError(f"Unknown action type pointer: {self.last_data_type}")

        Vs = safety_value(boundary_state[2:].cpu().numpy(), ENVELOPE_P) * 0.01

        # Batch size for PHY-Teacher Buffer
        tea_batch_size = max(min(L - 1, int(L * Vs)), 1)

        # Batch size for DRL-Student Buffer
        stu_batch_size = L - tea_batch_size
        
        stu_batch = self.stu_buf.sample(stu_batch_size) if stu_batch_size > 0 else None
        tea_batch = self.tea_buf.sample(tea_batch_size) if tea_batch_size > 0 else None

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

        return TransitionBatch(**merged_dict)
