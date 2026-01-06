#!/usr/bin/env python3
"""
DDPG Actor-Learner Implementation
Implements IMPALA-style distributed training for DDPG algorithm

Reference:
    - DDPG: Continuous control with deep reinforcement learning (ICLR 2016)
      https://arxiv.org/abs/1509.02971
    - IMPALA: Scalable Distributed Deep-RL (ICML 2018)
      https://arxiv.org/abs/1802.01561
"""

import torch
from rclpy.node import Node
from typing import Union, Tuple, Dict, Any

from edge_drl_student.common.logger import Logger
from edge_drl_student.distributed.actor_learner import ActorLearner
from edge_drl_student.agents.off_policy.ddpg_agent import DDPGAgent
from edge_drl_student.memory.buffers.off_policy.solo_replay_buffer import SoloReplayBuffer
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.memory.buffers.off_policy.cognitive_replay_buffer import CognitiveReplayBuffer


class DDPGActorLearner(ActorLearner):
    """
    DDPG-specific Actor-Learner implementation
    
    Architecture:
        Inference Agent (Fast):
            - Actor network (for action generation)
            - Critic network (optional, for debugging)
            - Target networks (synchronized from training agent)
        
        Training Agent (Slow):
            - Actor network (updated by policy gradient)
            - Critic network (updated by TD error)
            - Target Actor network (soft update)
            - Target Critic network (soft update)
    
    Synchronization:
        Every N updates, copy all networks from training agent to inference agent:
        - Actor → Inference Actor
        - Critic → Inference Critic
        - Target Actor → Inference Target Actor
        - Target Critic → Inference Target Critic
    
    Example:
        >>> # Create DDPG Actor-Learner
        >>> actor_learner = DDPGActorLearner(
        ...     obs_dim=24,
        ...     act_dim=12,
        ...     replay_buffer=buffer,
        ...     device='cuda:0',
        ...     logger=logger,
        ...     action_bound=(-10.0, 10.0),
        ...     sync_interval=5,
        ...     actor_lr=3e-4,
        ...     critic_lr=3e-4,
        ...     gamma=0.99,
        ...     tau=0.005
        ... )
        >>> 
        >>> # Generate action (10Hz)
        >>> action = actor_learner.act(observation)
        >>> 
        >>> # Update policy (2Hz)
        >>> losses = actor_learner.update(batch_size=256)
    """
    
    # Type hints for IDE autocomplete
    inference_agent: DDPGAgent
    training_agent: DDPGAgent
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        node: Node,
        replay_buffer: Union[SoloReplayBuffer, DualReplayBuffer, CognitiveReplayBuffer],
        device: str,
        logger: Logger,
        action_bound: Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor]],
        sync_interval: int = 5,
        log_interval: int = 100,
        **agent_kwargs
    ):
        """
        Initialize DDPG Actor-Learner
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            replay_buffer: Replay buffer instance
            device: PyTorch device ('cpu' or 'cuda:0')
            logger: Logger instance
            action_bound: Action bounds (min, max) - can be tuple of floats or tensors
            sync_interval: Synchronization interval (number of updates)
            **agent_kwargs: Additional DDPG agent arguments:
                - actor_lr (float): Actor learning rate (default: 3e-4)
                - critic_lr (float): Critic learning rate (default: 3e-4)
                - gamma (float): Discount factor (default: 0.99)
                - tau (float): Soft update coefficient (default: 0.005)
                - actor_hidden_dims (tuple): Actor hidden dimensions (default: (256, 256, 256))
                - critic_hidden_dims (tuple): Critic hidden dimensions (default: (256, 256, 256))
                - noise (Noise): Noise process (default: OUNoise)
                - reward_normalization (bool): Enable reward normalization (default: True)
                - reward_clip_range (float): Reward clip range (default: 10.0)
        
        Example:
            >>> actor_learner = DDPGActorLearner(
            ...     obs_dim=24,
            ...     act_dim=12,
            ...     replay_buffer=buffer,
            ...     device='cuda:0',
            ...     logger=logger,
            ...     action_bound=(-10.0, 10.0),
            ...     sync_interval=5,
            ...     actor_lr=3e-4,
            ...     critic_lr=3e-4,
            ...     gamma=0.99,
            ...     tau=0.005
            ... )
        """
        # Call parent constructor
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            node=node,
            replay_buffer=replay_buffer,
            device=device,
            logger=logger,
            action_bound=action_bound,
            sync_interval=sync_interval,
            **agent_kwargs
        )
        
        # DDPG-specific statistics
        self.total_actor_loss = 0.0
        self.total_critic_loss = 0.0
        
        # Log DDPG-specific info
        self.log_interval = log_interval 
        self._log_ddpg_info()
    
    # ==================== Abstract Method Implementation ====================
    
    def _create_agent(self) -> DDPGAgent:
        """
        Create DDPG agent instance
        
        Returns:
            DDPGAgent instance with specified configuration
        """
        return DDPGAgent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            replay_buffer=self.replay_buffer,
            action_bound=self.action_bound,
            device=self.device,
            **self.agent_kwargs
        )
    
    def _sync_agent_parameters(self):
        """
        Synchronize DDPG-specific parameters
        
        Copies all network parameters from training agent to inference agent:
        1. Actor network (policy)
        2. Critic network (Q-function)
        3. Target Actor network
        4. Target Critic network
        
        Note:
            This method is called within a thread lock in the parent class,
            so thread safety is already guaranteed.
        """
        # 1. Copy Actor network parameters
        self.inference_agent.actor.load_state_dict(
            self.training_agent.actor.state_dict()
        )
        
        # 2. Copy Critic network parameters
        self.inference_agent.critic.load_state_dict(
            self.training_agent.critic.state_dict()
        )
        
        # 3. Copy Target Actor network parameters
        self.inference_agent.target_actor.load_state_dict(
            self.training_agent.target_actor.state_dict()
        )
        
        # 4. Copy Target Critic network parameters
        self.inference_agent.target_critic.load_state_dict(
            self.training_agent.target_critic.state_dict()
        )
        
        # 5. Sync normalizers if they exist
        if hasattr(self.training_agent, 'actor_normalizer') and hasattr(self.inference_agent, 'actor_normalizer'):
            self.inference_agent.actor_normalizer.load_state_dict(
                self.training_agent.actor_normalizer.state_dict()
            )
        
        if hasattr(self.training_agent, 'critic_normalizer') and hasattr(self.inference_agent, 'critic_normalizer'):
            self.inference_agent.critic_normalizer.load_state_dict(
                self.training_agent.critic_normalizer.state_dict()
            )
        
        # 6. Sync reward normalizer if it exists
        if (hasattr(self.training_agent, 'reward_normalizer') and 
            self.training_agent.reward_normalizer is not None and
            hasattr(self.inference_agent, 'reward_normalizer') and
            self.inference_agent.reward_normalizer is not None):
            self.inference_agent.reward_normalizer.load_state_dict(
                self.training_agent.reward_normalizer.state_dict()
            )
    
    # ==================== DDPG-Specific Methods ====================
    
    def act(
        self,
        observation: torch.Tensor,
        noise_scale: float = 0.1,
        explore: bool = True
    ) -> torch.Tensor:
        """
        Generate action using inference agent
        
        Args:
            observation: Current observation [batch_size, obs_dim] or [obs_dim]
            noise_scale: Scale factor for exploration noise (default: 0.1)
            explore: If True, add exploration noise (default: True)
        
        Returns:
            Action tensor [batch_size, act_dim] or [act_dim]
        
        Example:
            >>> # During training (with exploration noise)
            >>> action = actor_learner.act(obs, noise_scale=0.1, explore=True)
            >>> 
            >>> # During evaluation (deterministic)
            >>> action = actor_learner.act(obs, explore=False)
        """
        return self.inference_agent.act(
            obs=observation,
            noise_scale=noise_scale,
            explore=explore
        )
    
    def update(self, batch_size: int) -> Dict[str, float]:
        """
        Update DDPG policy using training agent
        
        Args:
            batch_size: Batch size for training
        
        Returns:
            Dictionary of losses and metrics:
            {
                'actor_loss': float,      # Actor network loss
                'critic_loss': float,     # Critic network loss (TD error)
                'sync_time': float,       # Sync time (if synced)
            }
        
        Example:
            >>> losses = actor_learner.update(batch_size=256)
            >>> print(f"Actor loss: {losses['actor_loss']:.4f}")
            >>> print(f"Critic loss: {losses['critic_loss']:.4f}")
        """
        # Update using training agent (returns tuple)
        actor_loss, critic_loss = self.training_agent.update(batch_size)
        
        # Convert to dictionary
        losses = {
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
            'critic_loss': critic_loss.item() if isinstance(critic_loss, torch.Tensor) else critic_loss,
        }
        
        # Update statistics
        self.update_count += 1
        self.total_actor_loss += losses['actor_loss']
        self.total_critic_loss += losses['critic_loss']
        
        # Check if should synchronize parameters
        if self.should_sync(self.update_count):
            sync_time = self.sync_parameters()
            losses['sync_time'] = sync_time
        
        # Log DDPG-specific metrics
        self._log_ddpg_metrics(losses)
        
        return losses
    
    # ==================== Statistics ====================
    
    def get_ddpg_stats(self) -> Dict[str, Any]:
        """
        Get DDPG-specific statistics
        
        Returns:
            Dictionary containing:
            - avg_actor_loss: Average actor loss
            - avg_critic_loss: Average critic loss
            - update_count: Total number of updates
        
        Example:
            >>> stats = actor_learner.get_ddpg_stats()
            >>> print(f"Avg actor loss: {stats['avg_actor_loss']:.4f}")
        """
        if self.update_count == 0:
            return {
                'avg_actor_loss': 0.0,
                'avg_critic_loss': 0.0,
                'update_count': 0,
            }
        
        return {
            'avg_actor_loss': self.total_actor_loss / self.update_count,
            'avg_critic_loss': self.total_critic_loss / self.update_count,
            'update_count': self.update_count,
        }
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get network information
        
        Returns:
            Dictionary containing network parameter counts and info
        
        Example:
            >>> info = actor_learner.get_network_info()
            >>> print(f"Actor params: {info['actor_params']:,}")
            >>> print(f"Critic params: {info['critic_params']:,}")
        """
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'actor_params': count_parameters(self.training_agent.actor),
            'critic_params': count_parameters(self.training_agent.critic),
            'target_actor_params': count_parameters(self.training_agent.target_actor),
            'target_critic_params': count_parameters(self.training_agent.target_critic),
            'total_params': (
                count_parameters(self.training_agent.actor) +
                count_parameters(self.training_agent.critic) +
                count_parameters(self.training_agent.target_actor) +
                count_parameters(self.training_agent.target_critic)
            ),
        }
    
    # ==================== Logging ====================
    
    def _log_ddpg_info(self):
        """Log DDPG-specific initialization information"""
        network_info = self.get_network_info()
        
        self.node.get_logger().info(
            f"\n{'='*60}\n"
            f"🎯 DDPG Actor-Learner Configuration\n"
            f"{'='*60}\n"
            f"  Actor LR:             {self.agent_kwargs.get('actor_lr', 3e-4)}\n"
            f"  Critic LR:            {self.agent_kwargs.get('critic_lr', 3e-4)}\n"
            f"  Gamma:                {self.agent_kwargs.get('gamma', 0.99)}\n"
            f"  Tau:                  {self.agent_kwargs.get('tau', 0.005)}\n"
            f"  Actor Hidden Dims:    {self.agent_kwargs.get('actor_hidden_dims', (256, 256, 256))}\n"
            f"  Critic Hidden Dims:   {self.agent_kwargs.get('critic_hidden_dims', (256, 256, 256))}\n"
            f"  Reward Normalization: {self.agent_kwargs.get('reward_normalization', True)}\n"
            f"  Reward Clip Range:    {self.agent_kwargs.get('reward_clip_range', 10.0)}\n"
            f"\n"
            f"  Network Parameters:\n"
            f"    Actor:         {network_info['actor_params']:,}\n"
            f"    Critic:        {network_info['critic_params']:,}\n"
            f"    Target Actor:  {network_info['target_actor_params']:,}\n"
            f"    Target Critic: {network_info['target_critic_params']:,}\n"
            f"    Total:         {network_info['total_params']:,}\n"
            f"{'='*60}\n"
        )
    
    def _log_ddpg_metrics(self, losses: Dict[str, float]):
        """
        Log DDPG-specific metrics to TensorBoard
        
        Args:
            losses: Dictionary of losses from update
        """
        if not isinstance(losses, dict):
            return
        
        # Log to TensorBoard
        if 'actor_loss' in losses:
            self.logger.writer.add_scalar('Perf/actor_loss', losses['actor_loss'], self.update_count)
        
        if 'critic_loss' in losses:
            self.logger.writer.add_scalar('Perf/critic_loss', losses['critic_loss'], self.update_count)
        
        # Log average statistics 
        if self.update_count % self.log_interval == 0:
            stats = self.get_ddpg_stats()
            self.logger.writer.add_scalar('Perf/avg_actor_loss', stats['avg_actor_loss'], self.update_count)
            self.logger.writer.add_scalar('Perf/avg_critic_loss', stats['avg_critic_loss'], self.update_count)  
        
        # Log to console
        if self.update_count % self.log_interval == 0:
            self.node.get_logger().info(
                f"📊 Update #{self.update_count}: "
                f"Actor Loss: {losses.get('actor_loss', 0.0):.4f}, "
                f"Critic Loss: {losses.get('critic_loss', 0.0):.4f}"
            )
    
    # ==================== Utility Methods ====================
    
    def get_actor_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get actor network weights (for debugging/visualization)
        
        Returns:
            Dictionary of actor network parameters
        """
        return {
            name: param.detach().clone()
            for name, param in self.training_agent.actor.named_parameters()
        }
    
    def get_critic_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get critic network weights (for debugging/visualization)
        
        Returns:
            Dictionary of critic network parameters
        """
        return {
            name: param.detach().clone()
            for name, param in self.training_agent.critic.named_parameters()
        }
    
    def compute_q_value(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value for given observation-action pair
        
        Args:
            observation: Observation tensor [batch_size, obs_dim]
            action: Action tensor [batch_size, act_dim]
        
        Returns:
            Q-value tensor [batch_size, 1]
        
        Example:
            >>> obs = torch.randn(1, 24).to(device)
            >>> action = torch.randn(1, 12).to(device)
            >>> q_value = actor_learner.compute_q_value(obs, action)
            >>> print(f"Q-value: {q_value.item():.2f}")
        """
        with torch.no_grad():
            critic_input = self.inference_agent._critic_input(observation, action)
            return self.inference_agent.critic(critic_input)
    
    def update_reward_normalizer(self, reward: float):
        """
        Update reward normalizer during episode (for collection phase)
        
        Args:
            reward: Current step reward
        
        Example:
            >>> actor_learner.update_reward_normalizer(reward)
        """
        if (hasattr(self.training_agent, 'reward_normalizer') and 
            self.training_agent.reward_normalizer is not None):
            self.training_agent.reward_normalizer.update(reward)
    
    def finish_episode(self):
        """
        Finish episode for reward normalizer
        
        Example:
            >>> actor_learner.finish_episode()
        """
        if (hasattr(self.training_agent, 'reward_normalizer') and 
            self.training_agent.reward_normalizer is not None):
            self.training_agent.reward_normalizer.finish_episode()
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"DDPGActorLearner(\n"
            f"  obs_dim={self.obs_dim},\n"
            f"  act_dim={self.act_dim},\n"
            f"  action_bound={self.action_bound},\n"
            f"  sync_interval={self.sync_interval},\n"
            f"  device='{self.device}',\n"
            f"  actor_lr={self.agent_kwargs.get('actor_lr', 3e-4)},\n"
            f"  critic_lr={self.agent_kwargs.get('critic_lr', 3e-4)},\n"
            f"  gamma={self.agent_kwargs.get('gamma', 0.99)},\n"
            f"  tau={self.agent_kwargs.get('tau', 0.005)},\n"
            f"  reward_normalization={self.agent_kwargs.get('reward_normalization', True)}\n"
            f")"
        )
