import torch
from typing import Callable, Dict, Generator, Tuple, Optional

from edge_drl_student.memory.buffers import SoloReplayBuffer
from edge_drl_student.memory.buffers.base_buffer import BaseBuffer
from edge_drl_student.memory.samplers.uniform_sampler import UniformSampler
from edge_drl_student.memory.buffers import Transition, TransitionBatch


class DualReplayBuffer(BaseBuffer):
    def __init__(self, 
                 student_buf_size: int, 
                 teacher_buf_size: int, 
                 sampler: Optional[UniformSampler] = None,
                 obs_dim: int = 12,
                 action_dim: int = 6,
                 device: str = "cpu",
                 ) -> None:

        if sampler is None:
            sampler = UniformSampler(max(student_buf_size, teacher_buf_size))

        self.student_buf_size = student_buf_size
        self.teacher_buf_size = teacher_buf_size
        self.device = device

        # DRL-Student Buffer
        self.stu_buf = SoloReplayBuffer(buffer_size=self.student_buf_size, 
                                        obs_dim=obs_dim, 
                                        action_dim=action_dim,
                                        sampler=sampler,
                                        device=self.device)

        # PHY-Teacher Buffer
        self.tea_buf = SoloReplayBuffer(buffer_size=self.teacher_buf_size, 
                                        obs_dim=obs_dim, 
                                        action_dim=action_dim,
                                        sampler=sampler,
                                        device=self.device)

        self.last_data_type = None


    @property
    def stu_buf_size(self) -> int:
        return self.stu_buf.size

    @property
    def tea_buf_size(self) -> int:
        return self.tea_buf.size

    def append(self, dataset: Dataset, action_type=None) -> None:
        """Appends a dataset of transitions to the storage.

        Args:
            dataset (Dataset): The dataset of transitions.
        """

        if torch.any(action_type == 1):
            self.tea_storage.append(dataset=dataset, action_type=action_type)
            self.last_data_type = ActionMode.TEACHER

        elif torch.any(action_type == 0):
            self.stu_storage.append(dataset=dataset, action_type=action_type)
            self.last_data_type = ActionMode.STUDENT

        else:
            raise RuntimeError(f"Unrecognized action type: {action_type} for dataset")

    def batch_generator(self, batch_size: int, batch_count: int) -> Generator[Transition, None, None]:
        """Returns a generator that yields batches of transitions.

        Args:
            batch_size (int): The size of the batches.
            batch_count (int): The number of batches to yield.
        Returns:
            A generator that yields batches of transitions.
        """
        L = batch_size
        # import pdb
        # pdb.set_trace()
        #######################   Replay Buffer Batch Sample   #######################

        # Get last row of DRL-Student Buffer for computing safety status
        if self.last_data_type == ActionMode.STUDENT:
            idx = self.stu_storage.sample_count - 1
            boundary_state = self.stu_storage._data['actor_observations'][idx]

        # Get last row of PHY-Teacher Buffer for computing safety status
        elif self.last_data_type == ActionMode.TEACHER:
            idx = self.tea_storage.sample_count - 1
            boundary_state = self.tea_storage._data['actor_observations'][idx]
        else:
            raise RuntimeError(f"Unknown action type pointer: {self.last_data_type}")

        # Calculate the safety status indicator by sT*P*s
        Vs = energy_value(boundary_state[2:].cpu().numpy(), ENVELOPE_P) * 0.01
        # print(f"Vs: {Vs}")

        # Batch size for PHY-Teacher Buffer
        tea_batch_size = max(min(L - 1, int(L * Vs)), 1)

        # Batch size for DRL-Student Buffer
        # stu_batch_size = max(L - min(L, int(L * Vs)), 1)
        stu_batch_size = L - tea_batch_size

        stu_gen = self.stu_storage.batch_generator(stu_batch_size, batch_count) if self.stu_transition_count > 0 else None
        tea_gen = self.tea_storage.batch_generator(tea_batch_size, batch_count) if self.tea_transition_count > 0 else None

        for _ in range(batch_count):
            batches = []
            if stu_gen:
                batches.append(next(stu_gen))
            if tea_gen:
                batches.append(next(tea_gen))
            # stu_batch = next(stu_gen)
            # tea_batch = next(tea_gen)

            # Merge two batches
            merged_batch = {
                k: torch.cat([b[k] for b in batches if k in b], dim=0)
                for k in (batches[0].keys() if batches else [])
            }

            yield merged_batch

    @property
    def sample_count(self) -> int:
        """Returns the number of individual transitions stored in the storage."""
        return self.stu_transition_count + self.tea_transition_count