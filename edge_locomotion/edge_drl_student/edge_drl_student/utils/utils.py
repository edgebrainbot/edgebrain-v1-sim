import enum
import numpy as np

class ActionMode(enum.Enum):
    STUDENT = 0
    TEACHER = 1
    UNCERTAIN = 2
    
    
def safety_value(state: np.ndarray, p_mat: np.ndarray) -> float:
    """
    Get system safety value represented by s^T @ P @ s
    """
    safety_val = state.transpose() @ p_mat @ state
    return safety_val.item()