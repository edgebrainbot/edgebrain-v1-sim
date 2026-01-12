#!/usr/bin/env python3
"""
Episode Manager Module
Manages episode lifecycle, statistics, and logging for RL training

Features:
    ✅ Episode lifecycle management (start, step, end)
    ✅ Reward tracking (total, components, moving average)
    ✅ Episode length tracking
    ✅ Action mode statistics (student vs teacher)
    ✅ Success/failure tracking
    ✅ Automatic timeout detection
    ✅ TensorBoard logging
    ✅ Comprehensive statistics
"""

import time
import numpy as np
from rclpy.node import Node
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from edge_drl_student.common.logger import Logger
from edge_drl_student.utils.utils import ActionMode

# ==================== Episode Statistics ====================
@dataclass
class EpisodeStats:
    """
    Episode Statistics Data Class
    
    Attributes:
        episode_id: Episode number
        total_reward: Total reward accumulated
        episode_length: Number of steps
        success: Whether episode was successful
        reason: Episode termination reason
        duration: Episode duration in seconds
        reward_components: Dictionary of reward components
        student_action_count: Number of student actions
        teacher_action_count: Number of teacher actions
    """
    episode_id: int
    total_reward: float
    episode_length: int
    success: bool
    reason: str
    duration: float
    reward_components: Dict[str, float] = field(default_factory=dict)
    student_action_count: int = 0
    teacher_action_count: int = 0
    
    @property
    def activation_ratio(self) -> float:
        """Teacher activation ratio (0.0 to 1.0)"""
        total = self.student_action_count + self.teacher_action_count
        return self.teacher_action_count / total if total > 0 else 0.0
    
    @property
    def average_reward(self) -> float:
        """Average reward per step"""
        return self.total_reward / self.episode_length if self.episode_length > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'episode_id': self.episode_id,
            'total_reward': self.total_reward,
            'episode_length': self.episode_length,
            'success': self.success,
            'reason': self.reason,
            'duration': self.duration,
            'activation_ratio': self.activation_ratio,
            'average_reward': self.average_reward,
            'reward_components': self.reward_components,
            'student_action_count': self.student_action_count,
            'teacher_action_count': self.teacher_action_count,
        }


# ==================== Episode Manager ====================
class EpisodeManager:
    """
    Episode Manager for RL Training
    
    Manages episode lifecycle, tracks statistics, and logs to TensorBoard.
    
    Architecture:
        ┌─────────────────────────────────────────────────┐
        │                 Episode Manager                 │
        ├─────────────────────────────────────────────────┤
        │  ┌──────────────────────────────────────────┐   │
        │  │  Episode Lifecycle                       │   │
        │  │  - start_episode()                       │   │
        │  │  - update_step()                         │   │
        │  │  - end_episode()                         │   │
        │  └──────────────────────────────────────────┘   │
        │                                                 │
        │  ┌──────────────────────────────────────────┐   │
        │  │  Statistics Tracking                     │   │
        │  │  - Total reward                          │   │
        │  │  - Reward components                     │   │
        │  │  - Episode length                        │   │
        │  │  - Action counts                         │   │
        │  │  - Success/failure                       │   │
        │  └──────────────────────────────────────────┘   │
        │                                                 │
        │  ┌──────────────────────────────────────────┐   │
        │  │  Moving Statistics                       │   │
        │  │  - Moving average reward                 │   │
        │  │  - Moving average length                 │   │
        │  │  - Success rate                          │   │
        │  └──────────────────────────────────────────┘   │
        │                                                 │
        │  ┌──────────────────────────────────────────┐   │
        │  │  TensorBoard Logging                     │   │
        │  │  - Episode metrics                       │   │
        │  │  - Reward components                     │   │
        │  │  - Moving averages                       │   │
        │  └──────────────────────────────────────────┘   │
        └─────────────────────────────────────────────────┘
    
    Example:
        >>> # Create episode manager
        >>> config = EpisodeConfig(max_episode_length=1000)
        >>> manager = EpisodeManager(
        ...     node=node,
        ...     logger=logger,
        ...     config=config
        ... )
        >>> 
        >>> # Episode lifecycle
        >>> manager.start_episode()
        >>> 
        >>> # During episode
        >>> for step in range(1000):
        ...     reward_components = {
        ...         'obstacle': -0.1,
        ...         'goal_distance': 0.5,
        ...         'body_state': -0.05
        ...     }
        ...     total_reward = manager.update_step(reward_components)
        ...     manager.increment_action_count(is_student=True)
        ...     
        ...     if done:
        ...         break
        >>> 
        >>> # End episode
        >>> stats, activation_ratio = manager.log_episode_end()
        >>> print(f"Episode {stats['episode_id']} done: {stats['reward']:.2f}")
    """
    
    def __init__(
        self,
        logger: Logger,
        max_episode_length: int = 1000,
        training_episode: int = 10000,
        log_interval: int = 10,
        moving_average_window: int = 100,
    ) -> None:
        """
        Initialize Episode Manager
        
        Args:
            node: ROS2 node instance
            logger: Logger instance
            config: Episode configuration (default: EpisodeConfig())
        """
        self.logger: Logger = logger
        
        # ========== Episode State ==========
        self.episode_count: int = 0
        self.current_episode_length: int = 0
        self.current_episode_reward: float = 0.0
        self.current_episode_start_time: float = 0.0

        self.log_interval: int = log_interval
        self.max_episode_length: int = max_episode_length
        self.training_episode: int = training_episode
        self.moving_average_window: int = moving_average_window
        
        # ========== Reward Components Tracking ==========
        self.current_reward_components: Dict[str, float] = {}
        self.cumulative_reward_components: Dict[str, float] = {}
        
        # ========== Action Counting ==========
        self.stu_action_count: int = 0
        self.tea_action_count: int = 0
        
        # ========== Episode History ==========
        self.episode_history: List[EpisodeStats] = []
        
        # ========== Moving Statistics ==========
        self.reward_window: deque = deque(maxlen=self.moving_average_window)
        self.length_window: deque = deque(maxlen=self.moving_average_window)
        self.success_window: deque = deque(maxlen=self.moving_average_window)
        
        # ========== Global Statistics ==========
        self.total_steps: int = 0
        self.total_reward: float = 0.0
        self.success_count: int = 0
        self.failure_count: int = 0
        self.timeout_count: int = 0
        
        print(f"\n{'='*70}\n"
              f"✅ Episode Manager initialized\n"
              f"{'='*70}\n"
              f"⚙️  Configuration:\n"
              f"   Max Episode Length:        {self.max_episode_length}\n"
              f"   Training Episodes:         {self.training_episode}\n"
              f"   Moving Average Window:     {self.moving_average_window}\n"
              f"   Log Interval:              {self.log_interval}\n"
              f"{'='*70}"
              )
    
    # ==================== Episode Lifecycle ====================    
    def reset_episode(self) -> None:
        """
        Reset current episode state without incrementing episode count
        
        Example:
            >>> manager.reset_episode()
            >>> # Current episode is reset
        """
        self.current_episode_length = 0
        self.current_episode_reward = 0.0
        self.current_episode_start_time = time.time()
        
        # Reset reward components
        self.current_reward_components = {}
        self.cumulative_reward_components = {}
        
        # Reset action counts
        self.stu_action_count = 0
        self.tea_action_count = 0
        
        print(
            f"\n{'='*70}\n"
            f"🔄 Episode {self.episode_count} reset\n"
            f"{'='*70}"
        )
    
    def update(self, reward_components: Dict[str, float], action_mode: ActionMode):
        """
        Update episode with new step
        
        Args:
            reward_components: Dictionary of reward components
                Example: {'obstacle': -0.1, 'goal_distance': 0.5, 'body_state': -0.05}
        
        Returns:
            float: Total reward for this step
        
        Example:
            >>> reward_components = {
            ...     'obstacle': -0.1,
            ...     'goal_distance': 0.5,
            ...     'body_state': -0.05
            ... }
            >>> total_reward = manager.update(reward_components, action_mode)
            >>> print(f"Step reward: {total_reward:.3f}")
        """
        
        # Calculate total reward
        total_reward = sum(reward_components.values())
        
        # Update episode state
        self.current_episode_length += 1
        self.current_episode_reward += total_reward
        self.total_steps += 1
        self.total_reward += total_reward
        
        # Update reward components
        self.current_reward_components = reward_components.copy()
        for key, value in reward_components.items():
            self.cumulative_reward_components[key] = (
                self.cumulative_reward_components.get(key, 0.0) + value
            )
        
        self.increment_action_count(action_mode=action_mode)
    
    
    def increment_action_count(self, action_mode: ActionMode) -> None:
        """
        Increment action counter
        
        Args:
            action_mode: ActionMode enum indicating student or teacher action
        
        Example:
            >>> manager.increment_action_count(action_mode=ActionMode.STUDENT)
            >>> manager.increment_action_count(action_mode=ActionMode.TEACHER)
        """
        if action_mode == ActionMode.STUDENT:
            self.stu_action_count += 1
            
        elif action_mode == ActionMode.TEACHER:
            self.tea_action_count += 1
        else:
            raise ValueError(f"Unknown ActionMode: {action_mode}")
    
    def is_episode_done(self) -> bool:
        """
        Check if episode should end (timeout)
        
        Returns:
            bool: True if episode length exceeds max_episode_length
        
        Example:
            >>> if manager.is_episode_done():
            ...     manager.log_episode_end()
        """
        return self.current_episode_length >= self.max_episode_length
    
    def should_continue_training(self) -> bool:
        """
        Check if training should continue
        
        Returns:
            bool: True if more episodes should be trained
        
        Example:
            >>> while manager.should_continue_training():
            ...     # Run episode
            ...     manager.update(...)
            ...     if done:
            ...         manager.log_episode_end(...)
        """
        return self.episode_count < self.training_episode

    def log_episode_end(self, success: Optional[bool], reason: str = 'unknown') -> Dict:
        """
        End episode and log statistics
        
        Args:
            success: Whether episode was successful
            reason: Episode termination reason
        
        Returns:
            Dict: Episode statistics dictionary
        
        Example:
            >>> stats = manager.log_episode_end(
            ...     success=True,
            ...     reason='goal_reached'
            ... )
            >>> print(f"Episode reward: {stats['reward']:.2f}")
            >>> print(f"Activation ratio: {stats['activation_ratio']:.2%}")
        """
        
        # Calculate episode duration
        episode_duration = time.time() - self.current_episode_start_time
        
        # Auto-detect reason if unknown
        if reason == 'unknown':
            if self.is_episode_done():
                reason = 'timeout'
            elif success:
                reason = 'success'
            else:
                reason = 'failure'
        
        # Update global statistics
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        if reason == 'timeout':
            self.timeout_count += 1
        
        # Create episode stats
        stats = EpisodeStats(
            episode_id=self.episode_count,
            total_reward=self.current_episode_reward,
            episode_length=self.current_episode_length,
            success=success,
            reason=reason,
            duration=episode_duration,
            reward_components=self.cumulative_reward_components.copy(),
            student_action_count=self.stu_action_count,
            teacher_action_count=self.tea_action_count,
        )
        
        # Add to history
        self.episode_history.append(stats)
        
        # Update moving statistics
        self.reward_window.append(self.current_episode_reward)
        self.length_window.append(self.current_episode_length)
        self.success_window.append(1.0 if success else 0.0)
        
        # Log to TensorBoard
        self._log_episode_to_tensorboard(stats)
        
        # Console log (every N episodes)
        if self.episode_count % self.log_interval == 0:
            self._print_episode_summary()
        
        # Increment episode count
        self.episode_count += 1
        
        return stats.to_dict()
    
    # ==================== Statistics ====================
    
    def get_moving_average_reward(self) -> float:
        """Get moving average reward"""
        return np.mean(self.reward_window) if len(self.reward_window) > 0 else 0.0
    
    def get_moving_average_length(self) -> float:
        """Get moving average episode length"""
        return np.mean(self.length_window) if len(self.length_window) > 0 else 0.0
    
    def get_success_rate(self) -> float:
        """Get success rate (0.0 to 1.0)"""
        return np.mean(self.success_window) if len(self.success_window) > 0 else 0.0
    
    def get_global_statistics(self) -> Dict:
        """
        Get global statistics
        
        Returns:
            Dict: Global statistics including:
                - episode_count: Total episodes
                - total_steps: Total steps across all episodes
                - total_reward: Total reward across all episodes
                - success_count: Number of successful episodes
                - failure_count: Number of failed episodes
                - timeout_count: Number of timeout episodes
                - success_rate: Overall success rate
                - average_reward: Average reward per episode
                - average_length: Average episode length
        """
        return {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'timeout_count': self.timeout_count,
            'success_rate': self.success_count / self.episode_count if self.episode_count > 0 else 0.0,
            'average_reward': self.total_reward / self.episode_count if self.episode_count > 0 else 0.0,
            'average_length': self.total_steps / self.episode_count if self.episode_count > 0 else 0.0,
            'moving_average_reward': self.get_moving_average_reward(),
            'moving_average_length': self.get_moving_average_length(),
            'moving_average_success_rate': self.get_success_rate(),
        }
    
    def get_current_episode_info(self) -> Dict:
        """
        Get current episode information
        
        Returns:
            Dict: Current episode state
        """
        total_actions = self.stu_action_count + self.tea_action_count
        teacher_activation_ratio = self.tea_action_count / total_actions if total_actions > 0 else 0.0
    
        return {
            'episode_id': self.episode_count,
            'length': self.current_episode_length,
            'reward': self.current_episode_reward,
            'duration': time.time() - self.current_episode_start_time,
            'student_action_count': self.stu_action_count,
            'teacher_action_count': self.tea_action_count,
            'teacher_activation_ratio': teacher_activation_ratio,
            'reward_components': self.cumulative_reward_components.copy(),
        }
    
    def get_recent_episodes(self, n: int = 10) -> List[Dict]:
        """
        Get recent episode statistics
        
        Args:
            n: Number of recent episodes to return
        
        Returns:
            List[Dict]: List of recent episode statistics
        """
        return [
            stats.to_dict() 
            for stats in self.episode_history[-n:]
        ]
    
    # ==================== Logging ====================
    def _log_episode_to_tensorboard(self, stats: EpisodeStats) -> None:
        """Log episode statistics to TensorBoard"""
        episode_id = stats.episode_id
        
        # Episode metrics
        self.logger.writer.add_scalar('Episode/reward', stats.total_reward, episode_id)
        self.logger.writer.add_scalar('Episode/length', stats.episode_length, episode_id)
        self.logger.writer.add_scalar('Episode/duration', stats.duration, episode_id)
        self.logger.writer.add_scalar('Episode/success', 1.0 if stats.success else 0.0, episode_id)
        self.logger.writer.add_scalar('Episode/teacher_activation_ratio', stats.teacher_activation_ratio, episode_id)
        self.logger.writer.add_scalar('Episode/average_reward', stats.average_reward, episode_id)
        
        # Reward components
        for component, value in stats.reward_components.items():
            self.logger.writer.add_scalar(f'Episode/reward_{component}', value, episode_id)
        
        # Moving averages
        self.logger.writer.add_scalar('Episode/moving_avg_reward', self.get_moving_average_reward(), episode_id)
        self.logger.writer.add_scalar('Episode/moving_avg_length', self.get_moving_average_length(), episode_id)    
        self.logger.writer.add_scalar('Episode/success_rate', self.get_success_rate(), episode_id)  
        
        # Action counts
        self.logger.writer.add_scalar('Episode/student_action_count', stats.student_action_count, episode_id)
        self.logger.writer.add_scalar('Episode/teacher_action_count', stats.teacher_action_count, episode_id)
    
    def _print_episode_summary(self) -> None:
        """Print episode summary to console"""
        global_stats = self.get_global_statistics()
        
        print(
            f"\n{'='*70}\n"
            f"📊 Episode Summary (Last {self.log_interval} episodes)\n"
            f"{'='*70}\n"
            f"Episodes: {global_stats['episode_count']}\n"
            f"  Total Steps:         {global_stats['total_steps']}\n"
            f"  Total Reward:        {global_stats['total_reward']:.2f}\n"
            f"  Average Reward:      {global_stats['average_reward']:.2f}\n"
            f"  Average Length:      {global_stats['average_length']:.1f}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"Moving Statistics (Window: {self.moving_average_window}):\n"
            f"  Avg Reward:          {global_stats['moving_average_reward']:.2f}\n"
            f"  Avg Length:          {global_stats['moving_average_length']:.1f}\n"
            f"  Success Rate:        {global_stats['moving_average_success_rate']:.2%}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"Overall Statistics:\n"
            f"  Success:             {global_stats['success_count']}\n"
            f"  Failure:             {global_stats['failure_count']}\n"
            f"  Timeout:             {global_stats['timeout_count']}\n"
            f"  Success Rate:        {global_stats['success_rate']:.2%}\n"
            f"{'='*70}"
        )
    
    def print_current_episode(self) -> None:
        """Print current episode information"""
        info: Dict = self.get_current_episode_info()
        
        print(
            f"\n{'='*70}\n"
            f"📍 Current Episode {info['episode_id']}\n"
            f"{'='*70}\n"
            f"Length:              {info['length']}\n"
            f"Reward:              {info['reward']:.2f}\n"
            f"Duration:            {info['duration']:.2f}s\n"
            f"Student Actions:     {info['student_action_count']}\n"
            f"Teacher Actions:     {info['teacher_action_count']}\n"
            f"Teacher Activation Ratio:    {info['teacher_activation_ratio']:.2%}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"Reward Components:\n"
        )
        rew_components: Dict = info['reward_components']
        
        for component, value in rew_components.items():
            print(f"  {component:20s}: {value:8.3f}")
        
        print(f"{'='*70}")
    
    # ==================== Reset ====================
    def reset_statistics(self) -> None:
        """Reset all statistics (keep episode history)"""
        self.total_steps = 0
        self.total_reward = 0.0
        self.success_count = 0
        self.failure_count = 0
        self.timeout_count = 0
        
        self.reward_window.clear()
        self.length_window.clear()
        self.success_window.clear()
        
        print("📊 Episode statistics reset")
    
    def clear_history(self) -> None:
        """Clear episode history"""
        self.episode_history.clear()
        print("🗑️  Episode history cleared")
    
    def __repr__(self) -> str:
        """String representation"""
        global_stats = self.get_global_statistics()
        return (
            f"EpisodeManager(\n"
            f"  episode_count={global_stats['episode_count']},\n"
            f"  total_steps={global_stats['total_steps']},\n"
            f"  success_rate={global_stats['success_rate']:.2%},\n"
            f"  moving_avg_reward={global_stats['moving_average_reward']:.2f},\n"
            f")"
        )
