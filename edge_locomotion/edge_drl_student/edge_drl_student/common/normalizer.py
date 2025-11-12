import abc
import torch
import torch.nn as nn


# ============================================================
# ====                  Abstract Base Class               ====
# ============================================================
class BaseNormalizer(abc.ABC):
    """Abstract base class for all normalizers."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    @abc.abstractmethod
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized x."""
        pass

    @abc.abstractmethod
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Return x back to original scale."""
        pass

    def update(self, x: torch.Tensor):
        """Optional: update normalization statistics."""
        pass

    def to(self, device):
        """Move tensors to device."""
        self.device = device
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    def save(self, path: str):
        """Save normalization statistics to file."""
        torch.save(self.__dict__, path)

    def load(self, path: str):
        """Load normalization statistics from file."""
        data = torch.load(path, map_location=self.device)
        self.__dict__.update(data)


# ============================================================
# ======               Min–Max Normalizer               ======
# ============================================================
class MinMaxNormalizer(BaseNormalizer):
    """Linearly scales inputs from [min, max] to [-1, 1]."""

    def __init__(self, min_val, max_val, device="cpu"):
        super().__init__(device)
        self.min = torch.as_tensor(min_val, dtype=torch.float32, device=device)
        self.max = torch.as_tensor(max_val, dtype=torch.float32, device=device)

    def normalize(self, x):
        return 2 * (x - self.min) / (self.max - self.min + 1e-8) - 1.0

    def denormalize(self, x):
        return (x + 1.0) / 2.0 * (self.max - self.min) + self.min


# =============================================================
# ===   Running Mean / Std Normalizer (like VecNormalize)  ====
# =============================================================
class RunningMeanStdNormalizer(BaseNormalizer):
    """Online z-score normalization with running mean/std."""

    def __init__(self, shape, momentum=0.99, device="cpu"):
        super().__init__(device)
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(1e-4, dtype=torch.float32, device=device)
        self.momentum = momentum

    def update(self, x: torch.Tensor):
        x = x.detach()
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)

    def denormalize(self, x):
        return x * torch.sqrt(self.var + 1e-8) + self.mean


# =============================================================
# =====      Empirical Normalizer (nn.Module style)       =====
# =============================================================
class EmpiricalNormalizer(BaseNormalizer, nn.Module):
    """Learn mean/std online like a normalization layer."""

    def __init__(self, shape, eps=1e-6, until=None, device="cpu"):
        BaseNormalizer.__init__(self, device)
        nn.Module.__init__(self)
        self.eps = eps
        self.until = until

        self.register_buffer("_mean", torch.zeros(shape, device=device))
        self.register_buffer("_var", torch.ones(shape, device=device))
        self.register_buffer("_std", torch.ones(shape, device=device))
        self.count = 0

    def forward(self, x):
        if self.training:
            self.update(x)
        return self.normalize(x)

    def update(self, x: torch.Tensor):
        x = x.detach()
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / max(self.count, 1e-8)

        mean_x = torch.mean(x, dim=0)
        var_x = torch.var(x, dim=0, unbiased=False)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var + self.eps)

    def normalize(self, x):
        return (x - self._mean) / (self._std + self.eps)

    def denormalize(self, x):
        return x * (self._std + self.eps) + self._mean
