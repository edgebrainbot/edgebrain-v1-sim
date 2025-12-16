import abc
import torch
import numpy as np
import torch.nn as nn
from collections import deque


# ============================================================
# ====                  Abstract Base Class               ====
# ============================================================
class BaseNormalizer(abc.ABC):
    """Abstract base class for all normalizers."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    @abc.abstractmethod
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def update(self, x: torch.Tensor):
        """Optional override."""
        pass

    def to(self, device):
        self.device = device
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    def save(self, path: str):
        torch.save(self.__dict__, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.__dict__.update(data)


# ============================================================
# ======               Min–Max Normalizer               ======
# ============================================================
class MinMaxNormalizer(BaseNormalizer):
    """Linearly scales inputs from [min, max] → [-1, 1]."""

    def __init__(self, min_val, max_val, device="cpu"):
        super().__init__(device)
        self.min = torch.as_tensor(min_val, dtype=torch.float32, device=device)
        self.max = torch.as_tensor(max_val, dtype=torch.float32, device=device)

    def normalize(self, x):
        return 2 * (x - self.min) / (self.max - self.min + 1e-8) - 1.0

    def denormalize(self, x):
        return (x + 1.0) / 2.0 * (self.max - self.min) + self.min


# =============================================================
# ===   Running Mean / Std Normalizer (safe version)     ====
# =============================================================
class RunningMeanStdNormalizer(BaseNormalizer):
    """Online z-score normalization with no autograd involvement."""

    def __init__(self, shape, momentum=0.99, device="cpu"):
        super().__init__(device)
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(1e-4, dtype=torch.float32, device=device)
        self.momentum = momentum

    def update(self, x: torch.Tensor):
        with torch.no_grad():
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

            # Use copy_ to avoid creating new tensors (safe inplace on buffers)
            self.mean.copy_(new_mean)
            self.var.copy_(new_var)
            self.count.copy_(total_count)

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)

    def denormalize(self, x):
        return x * torch.sqrt(self.var + 1e-8) + self.mean


# =============================================================
# =====      Empirical Normalizer (SAFE nn.Module)       =====
# =============================================================
class EmpiricalNormalizer(BaseNormalizer, nn.Module):
    """Online normalization layer (safe, no autograd-side effects)."""

    def __init__(self, shape, eps=1e-6, until=None, device="cpu"):
        BaseNormalizer.__init__(self, device)
        nn.Module.__init__(self)

        self.eps = eps
        self.until = until

        # Running stats as buffers
        self.register_buffer("_mean", torch.zeros(shape, device=device))
        self.register_buffer("_var", torch.ones(shape, device=device))
        self.register_buffer("_std", torch.ones(shape, device=device))
        self.register_buffer("_count", torch.tensor(0.0, device=device))

    def forward(self, x):
        if self.training:
            self.update(x)
        return self.normalize(x)

    def update(self, x: torch.Tensor):
        with torch.no_grad():               # ********** KEY FIX **********
            x = x.detach()

            if self.until is not None and self._count.item() >= self.until:
                return

            batch_count = x.shape[0]

            # update count
            self._count.add_(batch_count)

            rate = batch_count / max(self._count.item(), 1e-8)

            mean_x = x.mean(dim=0)
            var_x = x.var(dim=0, unbiased=False)

            delta_mean = mean_x - self._mean
            new_mean = self._mean + rate * delta_mean
            new_var = self._var + rate * (var_x - self._var + delta_mean * (mean_x - new_mean))

            self._mean.copy_(new_mean)
            self._var.copy_(new_var)
            self._std.copy_(torch.sqrt(self._var + self.eps))

    def normalize(self, x):
        return (x - self._mean) / (self._std + self.eps)

    def denormalize(self, x):
        return x * (self._std + self.eps) + self._mean


class RewardNormalizerWindow:
    """基于滑动窗口的奖励归一化"""
    
    def __init__(self, window_size=1000, clip_range=5.0):
        """
        Args:
            window_size: 窗口大小
            clip_range: 裁剪范围
        """
        self.window_size = window_size
        self.clip_range = clip_range
        self.rewards = deque(maxlen=window_size)
        
    def normalize(self, reward):
        self.rewards.append(reward)
        
        if len(self.rewards) < 2:
            return reward
        
        mean = np.mean(self.rewards)
        std = np.std(self.rewards)
        
        if std > 1e-8:
            normalized_reward = (reward - mean) / std
        else:
            normalized_reward = reward - mean
        
        if self.clip_range is not None:
            normalized_reward = np.clip(
                normalized_reward,
                -self.clip_range,
                self.clip_range
            )
        
        return normalized_reward
