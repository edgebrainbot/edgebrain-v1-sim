import torch
import numpy as np
from abc import ABC, abstractmethod


# ============================== Base Class ==============================
class Noise(ABC):
    """Abstract noise generator that only outputs noise."""

    @abstractmethod
    def sample(self, shape) -> torch.Tensor:
        """Generate noise with given shape."""
        pass

    def reset(self):
        """Optional for stateful noise."""
        pass

    def __call__(self, shape):
        return self.sample(shape)


# ======================== Gaussian Noise ========================
class GaussianNoise(Noise):
    def __init__(self, sigma=0.1, device="cpu"):
        self.sigma = sigma
        self.device = device

    def sample(self, shape):
        return torch.randn(shape, device=self.device) * self.sigma


# ======================== Clipped Gaussian (TD3) ========================
class ClippedGaussianNoise(Noise):
    def __init__(self, sigma=0.2, clip=0.5, device="cpu"):
        self.sigma = sigma
        self.clip = clip
        self.device = device

    def sample(self, shape):
        noise = torch.randn(shape, device=self.device) * self.sigma
        return noise.clamp(-self.clip, self.clip)


# ============================== OU Noise ==============================
class OUNoise(Noise):
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, device="cpu"):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.device = device
        self.state = torch.ones(size, device=device) * mu

    def reset(self):
        self.state = torch.ones_like(self.state) * self.mu

    def sample(self, shape):
        dx = self.theta * (self.mu - self.state) + self.sigma * torch.randn_like(self.state)
        self.state += dx
        return self.state.clone()

    def __call__(self, shape):
        return self.sample(shape)


# ========================= Parameter Noise =========================
class ParameterNoise(Noise):
    """Noise applied to model parameters."""

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def apply(self, model):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * self.sigma)

    def sample(self, shape):
        return torch.zeros(shape)  # does not affect action


# ========================= Composite Noise =========================
class CompositeNoise(Noise):
    def __init__(self, *noises: Noise):
        self.noises = noises

    def sample(self, shape):
        result = torch.zeros(shape)
        for n in self.noises:
            result += n.sample(shape)
        return result


# ============================== Factory ==============================
class NoiseFactory:
    @staticmethod
    def create(name: str, **kwargs):
        name = name.lower()
        if name == "gaussian":
            return GaussianNoise(**kwargs)
        elif name == "clipped_gaussian":
            return ClippedGaussianNoise(**kwargs)
        elif name == "ou":
            return OUNoise(**kwargs)
        elif name == "parameter":
            return ParameterNoise(**kwargs)
        else:
            raise ValueError(f"Unknown noise type: {name}")
