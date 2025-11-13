import numpy as np
from .base_sampler import BaseSampler


class ACTRDeclarativeSampler(BaseSampler):
    """
    ACTRDeclarativeSampler
    ---------------
    A simple sampler that draws indices uniformly at random
    from the valid portion of the replay buffer.
    """

    def __init__(self, buffer_size: int):
        super().__init__(buffer_size)

    def fetch(self, batch_size: int, valid_size: int) -> np.ndarray:
        """
        Uniformly sample indices from [0, valid_size).

        Args:
            batch_size (int): Number of samples to fetch.
            valid_size (int): Number of valid transitions currently stored.

        Returns:
            np.ndarray[int]: Randomly sampled indices.
        """
        if valid_size <= 0:
            raise ValueError("Cannot fetch from an empty buffer.")
        if valid_size > self.buffer_size:
            raise ValueError(
                f"valid_size ({valid_size}) cannot exceed buffer_size ({self.buffer_size})."
            )

        indices = np.random.randint(0, valid_size, size=batch_size)
        return indices

    def update(self, indices: np.ndarray, **kwargs):
        """Uniform sampler has no internal state to update."""
        pass
