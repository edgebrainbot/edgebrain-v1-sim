import enum
import numpy as np

class ActionMode(enum.Enum):
    STUDENT = 0
    TEACHER = 1
    UNCERTAIN = 2
    
    
class ResetReason(enum.Enum):
    """
    Reset Reason Enumeration
    
    Defines all possible reasons for triggering a reset.
    
    Example:
        >>> reason = ResetReason.FALL_DETECTED
        >>> print(reason.name)  # 'FALL_DETECTED'
        >>> print(reason.value)  # 1
    """
    MANUAL = 0
    UNKNOWN = 1
    TIMEOUT = 2
    EXTERNAL = 3
    TASK_DONE = 4
    FALL_DETECTED = 5
    
    @classmethod
    def from_string(cls, reason_str: str) -> 'ResetReason':
        """
        Convert string to ResetReason enum
        
        Args:
            reason_str: Reason string (case-insensitive)
            
        Returns:
            ResetReason: Corresponding enum value
            
        Example:
            >>> reason = ResetReason.from_string('fall_detected')
            >>> assert reason == ResetReason.FALL_DETECTED
            >>> 
            >>> reason = ResetReason.from_string('TASK_DONE')
            >>> assert reason == ResetReason.TASK_DONE
        """
        reason_map = {
            'task_done': cls.TASK_DONE,
            'fall_detected': cls.FALL_DETECTED,
            'manual': cls.MANUAL,
            'timeout': cls.TIMEOUT,
            'external': cls.EXTERNAL,
            'unknown': cls.UNKNOWN,
        }
        
        # Try exact match first
        if reason_str in reason_map:
            return reason_map[reason_str]
        
        # Try case-insensitive match
        reason_lower = reason_str.lower()
        if reason_lower in reason_map:
            return reason_map[reason_lower]
        
        # Try enum name match
        try:
            return cls[reason_str.upper()]
        except KeyError:
            return cls.UNKNOWN
    
    def __str__(self) -> str:
        """String representation (lowercase with underscores)"""
        return self.name.lower()
    
    
def safety_value(state: np.ndarray, p_mat: np.ndarray) -> float:
    """
    Get system safety value represented by s^T @ P @ s
    """
    safety_val = state.transpose() @ p_mat @ state
    return safety_val.item()