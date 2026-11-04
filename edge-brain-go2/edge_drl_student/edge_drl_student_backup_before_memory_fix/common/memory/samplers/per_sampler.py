import torch
import numpy as np

from .base_sampler import BaseSampler


class PERSampler(BaseSampler):

    def __init__(
        self,
        replay_buf,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
        device: str = "cpu",
    ):
        self.replay_buf = replay_buf
        super().__init__(replay_buf.buffer_size)

        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.device = device

        # Every transition has one priority.
        self.priorities = torch.ones(
            replay_buf.buffer_size,
            dtype=torch.float32,
            device=device,
        )

        self.last_indices = None
        self.last_weights = None


    def fetch(
        self,
        plant_state: torch.Tensor,
        batch_size: int,
        valid_size: int,
    ):

        if valid_size <= 0:
            raise ValueError("Cannot fetch from an empty buffer.")

        # Only priorities corresponding to valid experiences.
        priorities = self.priorities[:valid_size]

        # PER:
        # P(i) = p_i^alpha / sum_j p_j^alpha
        probs = priorities.pow(self.alpha)
        probs = probs / probs.sum()

        replacement = valid_size < batch_size

        indices = torch.multinomial(
            probs,
            batch_size,
            replacement=replacement,
        )

        # Importance-sampling weights:
        # w_i = (N * P(i))^(-beta)
        weights = (
            valid_size * probs[indices]
        ).pow(-self.beta)

        weights = weights / weights.max()

        self.last_indices = indices
        self.last_weights = weights

        batch_samples = self.replay_buf.gather(indices)

        return batch_samples


    def update(self, indices, td_errors, **kwargs):

        if isinstance(indices, np.ndarray):
            indices = torch.as_tensor(
                indices,
                dtype=torch.long,
                device=self.device,
            )
        else:
            indices = indices.to(self.device)

        if not torch.is_tensor(td_errors):
            td_errors = torch.tensor(
                td_errors,
                dtype=torch.float32,
                device=self.device,
            )

        td_errors = td_errors.detach().abs().flatten()

        # p_i = |delta_i| + epsilon
        self.priorities[indices] = td_errors + self.eps


    def get_weights(self):
        return self.last_weights


    def get_indices(self):
        return self.last_indices