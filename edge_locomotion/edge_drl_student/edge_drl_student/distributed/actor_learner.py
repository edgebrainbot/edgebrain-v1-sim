#!/usr/bin/env python3
"""
Base Actor-Learner Architecture Module
Implements IMPALA-style distributed RL architecture

Reference:
    IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures
    https://arxiv.org/abs/1802.01561
"""

import time
import torch
import threading
from rclpy.node import Node
from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict, Any, Optional


from edge_drl_student.common.logger import Logger
from edge_drl_student.memory.buffers.base_buffer import BaseBuffer
from edge_drl_student.agents.off_policy_agent import OffPolicyAgent


class ActorLearner(ABC):
    
    """
    Abstract base class for Actor-Learner architecture
    
    Implements IMPALA-style distributed RL with two agents:
    - Inference Agent: Fast action generation (high frequency, e.g., 10Hz)
    - Training Agent: Slow policy learning (low frequency, e.g., 2Hz)
    - Periodic parameter synchronization from training to inference
    
    Architecture:
        ┌─────────────────┐
        │ Inference Agent │ ──> Fast action generation (10Hz)
        │   (Actor only)  │
        └────────┬────────┘
                 │ Sync parameters
                 │ (every N updates)
                 ↓
        ┌─────────────────┐
        │ Training Agent  │ ──> Slow policy learning (2Hz)
        │ (Actor + Critic)│
        └─────────────────┘
    
    Subclasses must implement:
        - _create_agent(): Create agent instance
        - _sync_agent_parameters(): Synchronize agent-specific parameters
    
    Example:
        >>> class DDPGActorLearner(ActorLearner):
        ...     def _create_agent(self):
        ...         return DDPGAgent(...)
        ...     
        ...     def _sync_agent_parameters(self):
        ...         self.inference_agent.actor.load_state_dict(
        ...             self.training_agent.actor.state_dict()
        ...         )
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        node: Node,
        replay_buffer: BaseBuffer,
        device: str,
        logger: Logger,
        action_bound: Tuple[float, float] = (-10.0, 10.0),
        sync_interval: int = 5,
        **agent_kwargs
    ):
        """
        Initialize Base Actor-Learner architecture
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            replay_buffer: Replay buffer instance (shared between agents)
            device: PyTorch device ('cpu' or 'cuda:0')
            logger: Logger instance for TensorBoard logging
            action_bound: Action bounds (min, max)
            sync_interval: Synchronization interval (number of updates)
            **agent_kwargs: Additional arguments passed to agent initialization
                           (e.g., learning_rate, gamma, tau, hidden_dim, etc.)
        """
        # Store configuration
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.node = node
        self.replay_buffer = replay_buffer
        self.device = device
        self.logger = logger
        self.action_bound = action_bound
        self.sync_interval = sync_interval
        self.agent_kwargs = agent_kwargs
        
        # Create agents (implemented by subclasses)
        self.inference_agent = self._create_agent()
        self.training_agent = self._create_agent()
        
        # Thread safety for parameter synchronization
        self.sync_lock = threading.Lock()
        
        # Synchronization statistics
        self.sync_count = 0
        self.last_sync_time = 0.0
        self.total_sync_time = 0.0
        self.avg_sync_time = 0.0
        
        # Update statistics
        self.update_count = 0
        
        # Log initialization
        self._log_initialization()
    
    # ==================== Abstract Methods ====================
    
    @abstractmethod
    def _create_agent(self) -> OffPolicyAgent:
        """
        Create agent instance (must be implemented by subclass)
        
        Returns:
            Agent instance (e.g., DDPGAgent, TD3Agent, SACAgent)
        
        Example:
            >>> def _create_agent(self):
            ...     return DDPGAgent(
            ...         obs_dim=self.obs_dim,
            ...         act_dim=self.act_dim,
            ...         replay_buffer=self.replay_buffer,
            ...         action_bound=self.action_bound,
            ...         device=self.device,
            ...         **self.agent_kwargs
            ...     )
        """
        raise NotImplementedError("Subclass must implement _create_agent()")
    
    @abstractmethod
    def _sync_agent_parameters(self):
        """
        Synchronize agent-specific parameters (must be implemented by subclass)
        
        This method should copy all necessary parameters from training_agent
        to inference_agent. Typically includes:
        - Actor network
        - Critic network(s)
        - Target networks
        - Other algorithm-specific parameters (e.g., log_alpha in SAC)
        
        Note:
            This method is called within a thread lock, so no need to
            worry about thread safety in the implementation.
        
        Example (DDPG):
            >>> def _sync_agent_parameters(self):
            ...     # Copy actor
            ...     self.inference_agent.actor.load_state_dict(
            ...         self.training_agent.actor.state_dict()
            ...     )
            ...     # Copy critic
            ...     self.inference_agent.critic.load_state_dict(
            ...         self.training_agent.critic.state_dict()
            ...     )
            ...     # Copy target networks
            ...     self.inference_agent.target_actor.load_state_dict(
            ...         self.training_agent.target_actor.state_dict()
            ...     )
            ...     self.inference_agent.target_critic.load_state_dict(
            ...         self.training_agent.target_critic.state_dict()
            ...     )
        
        Example (TD3):
            >>> def _sync_agent_parameters(self):
            ...     # Copy actor
            ...     self.inference_agent.actor.load_state_dict(
            ...         self.training_agent.actor.state_dict()
            ...     )
            ...     # Copy critic1 and critic2
            ...     self.inference_agent.critic1.load_state_dict(
            ...         self.training_agent.critic1.state_dict()
            ...     )
            ...     self.inference_agent.critic2.load_state_dict(
            ...         self.training_agent.critic2.state_dict()
            ...     )
            ...     # Copy target networks
            ...     self.inference_agent.target_actor.load_state_dict(
            ...         self.training_agent.target_actor.state_dict()
            ...     )
            ...     self.inference_agent.target_critic1.load_state_dict(
            ...         self.training_agent.target_critic1.state_dict()
            ...     )
            ...     self.inference_agent.target_critic2.load_state_dict(
            ...         self.training_agent.target_critic2.state_dict()
            ...     )
        """
        raise NotImplementedError("Subclass must implement _sync_agent_parameters()")
    
    # ==================== Core Methods ====================
    
    def sync_parameters(self) -> float:
        """
        Synchronize training network parameters to inference network
        
        This method is thread-safe and can be called from different threads.
        It copies all parameters from training_agent to inference_agent.
        
        Returns:
            float: Synchronization time in seconds
        
        Example:
            >>> sync_time = actor_learner.sync_parameters()
            >>> print(f"Sync took {sync_time*1000:.2f}ms")
        """
        sync_start = time.time()
        
        # Thread-safe parameter synchronization
        with self.sync_lock:
            self._sync_agent_parameters()
        
        # Update statistics
        sync_time = time.time() - sync_start
        self.last_sync_time = sync_time
        self.total_sync_time += sync_time
        self.sync_count += 1
        self.avg_sync_time = self.total_sync_time / self.sync_count
        
        # Log synchronization statistics
        self._log_sync_stats(sync_time)
        
        return sync_time
    
    def act(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Generate action using inference agent
        
        This method is used during environment interaction (high frequency).
        It uses the inference agent which has faster inference speed.
        
        Args:
            observation: Current observation tensor [batch_size, obs_dim] or [obs_dim]
            deterministic: Whether to use deterministic policy (default: False)
        
        Returns:
            Action tensor [batch_size, act_dim] or [act_dim]
        
        Example:
            >>> obs = torch.randn(1, obs_dim).to(device)
            >>> action = actor_learner.act(obs)
            >>> print(action.shape)  # [1, act_dim]
        """
        with torch.no_grad():
            if hasattr(self.inference_agent, 'act'):
                return self.inference_agent.act(observation, deterministic=deterministic)
            else:
                # Fallback: use actor directly
                return self.inference_agent.actor(observation)
    
    def update(self, batch_size: int) -> Dict[str, float]:
        """
        Update policy using training agent
        
        This method is used during training (low frequency).
        It samples a batch from replay buffer and updates the policy.
        
        Args:
            batch_size: Batch size for training
        
        Returns:
            Dictionary of losses and metrics, e.g.:
            {
                'actor_loss': float,
                'critic_loss': float,
                'q_value': float,
                ...
            }
        
        Example:
            >>> losses = actor_learner.update(batch_size=256)
            >>> print(f"Actor loss: {losses['actor_loss']:.4f}")
        """
        self.update_count += 1
        
        # Update using training agent
        losses = self.training_agent.update(batch_size)
        
        # Check if should synchronize parameters
        if self.should_sync(self.update_count):
            sync_time = self.sync_parameters()
            if isinstance(losses, dict):
                losses['sync_time'] = sync_time
        
        return losses
    
    def should_sync(self, update_count: Optional[int] = None) -> bool:
        """
        Check if should synchronize parameters
        
        Args:
            update_count: Current update count (default: use internal counter)
        
        Returns:
            bool: True if should synchronize
        
        Example:
            >>> if actor_learner.should_sync(update_count):
            ...     actor_learner.sync_parameters()
        """
        if update_count is None:
            update_count = self.update_count
        return update_count % self.sync_interval == 0
    
    # ==================== Model Management ====================
    
    def save_models(self, path: str):
        """
        Save training agent models to disk
        
        Args:
            path: Directory path to save models
        
        Example:
            >>> actor_learner.save_models('./models/checkpoint_1000')
        """
        self.training_agent.save_models(path)
        self.node.get_logger().info(f"✅ Models saved to: {path}")
    
    def load_models(self, path: str):
        """
        Load models from disk to both agents
        
        Args:
            path: Directory path to load models from
        
        Example:
            >>> actor_learner.load_models('./models/checkpoint_1000')
        """
        self.training_agent.load_models(path)
        self.inference_agent.load_models(path)
        self.node.get_logger().info(f"✅ Models loaded from: {path}")
    
    # ==================== Statistics ====================
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """
        Get synchronization statistics
        
        Returns:
            Dictionary containing:
            - sync_count: Total number of synchronizations
            - last_sync_time: Last synchronization time (seconds)
            - total_sync_time: Total synchronization time (seconds)
            - avg_sync_time: Average synchronization time (seconds)
            - sync_interval: Synchronization interval (updates)
        
        Example:
            >>> stats = actor_learner.get_sync_stats()
            >>> print(f"Avg sync time: {stats['avg_sync_time']*1000:.2f}ms")
        """
        return {
            'sync_count': self.sync_count,
            'last_sync_time': self.last_sync_time,
            'total_sync_time': self.total_sync_time,
            'avg_sync_time': self.avg_sync_time,
            'sync_interval': self.sync_interval,
        }
    
    def get_update_stats(self) -> Dict[str, Any]:
        """
        Get update statistics
        
        Returns:
            Dictionary containing:
            - update_count: Total number of updates
            - sync_count: Total number of synchronizations
            - updates_per_sync: Average updates per synchronization
        
        Example:
            >>> stats = actor_learner.get_update_stats()
            >>> print(f"Total updates: {stats['update_count']}")
        """
        return {
            'update_count': self.update_count,
            'sync_count': self.sync_count,
            'updates_per_sync': self.update_count / max(self.sync_count, 1),
        }
    
    # ==================== Logging ====================
    
    def _log_initialization(self):
        """Log initialization information"""
        self.node.get_logger().info(
            f"\n{'='*60}\n"
            f"🚀 {self.__class__.__name__} Initialized\n"
            f"{'='*60}\n"
            f"  Inference Agent: {self.inference_agent.__class__.__name__}\n"
            f"  Training Agent:  {self.training_agent.__class__.__name__}\n"
            f"  Observation Dim: {self.obs_dim}\n"
            f"  Action Dim:      {self.act_dim}\n"
            f"  Action Bound:    {self.action_bound}\n"
            f"  Sync Interval:   every {self.sync_interval} updates\n"
            f"  Device:          {self.device}\n"
            f"  Replay Buffer:   {self.replay_buffer.__class__.__name__}\n"
            f"{'='*60}\n"
        )
    
    def _log_sync_stats(self, sync_time: float):
        """
        Log synchronization statistics to TensorBoard
        
        Args:
            sync_time: Current synchronization time (seconds)
        """
        # Log to TensorBoard
        self.logger.writer.add_scalar('Perf/Sync/time_ms', sync_time * 1000, self.sync_count)
        self.logger.writer.add_scalar('Perf/Sync/cumulative_time_s', self.total_sync_time, self.sync_count) 
        self.logger.writer.add_scalar('Perf/Sync/avg_time_ms', self.avg_sync_time * 1000, self.sync_count)
        self.logger.writer.add_scalar('Perf/Sync/count', self.sync_count, self.update_count)
        
        # Log to console (every 10 syncs)
        if self.sync_count % 10 == 0:
            self.node.get_logger().info(
                f"🔄 Sync #{self.sync_count}: "
                f"{sync_time*1000:.2f}ms "
                f"(avg: {self.avg_sync_time*1000:.2f}ms)"
            )
    
    # ==================== Utility Methods ====================
    def set_train_mode(self):
        """Set both agents to training mode"""
        self.inference_agent.actor.train()
        self.training_agent.actor.train()
        if hasattr(self.training_agent, 'critic'):
            self.training_agent.critic.train()
    
    def set_eval_mode(self):
        """Set both agents to evaluation mode"""
        self.inference_agent.actor.eval()
        self.training_agent.actor.eval()
        if hasattr(self.training_agent, 'critic'):
            self.training_agent.critic.eval()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get agent information
        
        Returns:
            Dictionary containing agent information
        """
        return {
            'class_name': self.__class__.__name__,
            'inference_agent': self.inference_agent.__class__.__name__,
            'training_agent': self.training_agent.__class__.__name__,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'action_bound': self.action_bound,
            'sync_interval': self.sync_interval,
            'device': self.device,
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"{self.__class__.__name__}(\n"
            f"  inference_agent={self.inference_agent.__class__.__name__},\n"
            f"  training_agent={self.training_agent.__class__.__name__},\n"
            f"  obs_dim={self.obs_dim},\n"
            f"  act_dim={self.act_dim},\n"
            f"  sync_interval={self.sync_interval},\n"
            f"  device='{self.device}'\n"
            f")"
        )
