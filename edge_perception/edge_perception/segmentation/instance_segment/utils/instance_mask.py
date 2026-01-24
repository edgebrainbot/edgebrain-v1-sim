import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class InstanceMask:
    mask: np.ndarray  # bool [H, W]
    score: float = 1.0
    instance_id: Optional[int] = None