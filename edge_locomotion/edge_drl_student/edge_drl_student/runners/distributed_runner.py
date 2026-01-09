#!/usr/bin/env python3
"""
Distributed Agent Implementation with Configuration Classes
Implements IMPALA-style distributed training with integrated loops

Reference:
    - IMPALA: Scalable Distributed Deep-RL (ICML 2018)
      https://arxiv.org/abs/1802.01561
"""

from edge_drl_student.agents.off_policy.ddpg_agent import DDPGAgent
import torch
import time
import numpy as np
from rclpy.node import Node
from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any, Optional

from std_msgs.msg import Bool
from edge_msgs.msg import Transition, Vec12Msg, EdgeAction
from edge_drl_student.common.logger import Logger
from edge_drl_student.utils.utils import ActionMode, ResetReason
from edge_drl_student.common.reset_manager import ResetManager
from edge_drl_student.common.episode_handler import EpisodeManager
from edge_drl_student.agents.off_policy_agent import OffPolicyAgent
from edge_drl_student.agents.distributed.ddpg_actor_learner import DDPGActorLearner
from edge_drl_student.memory.buffers.off_policy.solo_replay_buffer import SoloReplayBuffer
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.memory.buffers.off_policy.cognitive_replay_buffer import CognitiveReplayBuffer
from edge_drl_student.processor.vision_processor import VisionProcessor

# ==================== Configuration Classes ====================

@dataclass
class TopicConfig:
    """ROS2 Topic Configuration"""
    transition: str = 'transition'
    tracking_error: str = 'tracking_error_stu'
    action: str = 'drl_student_action'
    learned_action: str = 'drl_student_learned_action'
    fall_detection: str = '/edge/is_fall' 


@dataclass
class LoopConfig:
    """Loop Frequency Configuration"""
    action_frequency: float = 10.0   # Hz
    learning_frequency: float = 2.0  # Hz


@dataclass
class TrainingConfig:
    """Training Configuration"""
    batch_size: int = 512
    save_interval: int = 500
    sync_interval: int = 5
    log_interval: int = 100
    model_save_path: str = './models'


@dataclass
class ResidualConfig:
    """Residual Control Configuration"""
    residual_gain: Optional[torch.Tensor] = None
    policy_blending_weight: float = 0.3
    
    def get_gain(self, act_dim: int, body_state_dim: int, device: str) -> torch.Tensor:
        """Get residual gain matrix"""
        if self.residual_gain is None:
            # Default: identity matrix
            return torch.eye(act_dim, body_state_dim, device=device)
        return self.residual_gain.to(device)


# ==================== Distributed Agent ====================
class DistributedRunner(DDPGActorLearner):
    """
    Distributed Agent with Configuration Classes and Fall Detection
    
    Architecture:
        ┌────────────────────────────────────┐
        │         DistributedRunner          │
        ├────────────────────────────────────┤
        │  ┌──────────────────────────────┐  │
        │  │  Acting Loop (10Hz)          │  │
        │  │  - Generate actions          │  │
        │  │  - Publish to robot          │  │
        │  │  - Use acting agent (fast)   │  │
        │  └──────────────────────────────┘  │
        │                                    │
        │  ┌──────────────────────────────┐  │
        │  │  Transition Callback         │  │
        │  │  - Collect experiences       │  │
        │  │  - Add to replay buffer      │  │
        │  │  - Update episode stats      │  │
        │  │  - Auto reset on done        │  │
        │  └──────────────────────────────┘  │
        │                                    │
        │  ┌──────────────────────────────┐  │
        │  │  Fall Detection Callback     │  │
        │  │  - Monitor /edge/is_fall     │  │
        │  │  - Detect rising edge        │  │
        │  │  - Trigger reset             │  │
        │  └──────────────────────────────┘  │
        │                                    │
        │  ┌──────────────────────────────┐  │
        │  │  Learning Loop (2Hz)         │  │
        │  │  - Sample from buffer        │  │
        │  │  - Update policy             │  │
        │  │  - Sync parameters           │  │
        │  │  - Use learning agent (slow) │  │
        │  └──────────────────────────────┘  │
        └────────────────────────────────────┘
    
    Features:
        ✅ Automatic transition collection
        ✅ Integrated acting loop (10Hz)
        ✅ Integrated learning loop (2Hz)
        ✅ Automatic parameter synchronization
        ✅ Episode management with auto reset
        ✅ Fall detection with auto reset
        ✅ Comprehensive logging
    
    Example:
        >>> # Create configurations
        >>> topic_cfg = TopicConfig()
        >>> loop_cfg = LoopConfig(action_frequency=10.0)
        >>> train_cfg = TrainingConfig(batch_size=512)
        >>> residual_cfg = ResidualConfig(policy_blending_weight=0.3)
        >>> 
        >>> # Create agent
        >>> agent = DistributedRunner(
        ...     obs_dim=36,
        ...     act_dim=6,
        ...     node=node,
        ...     replay_buffer=buffer,
        ...     device='cuda:0',
        ...     logger=logger,
        ...     vision_processor=vision_processor,
        ...     episode_manager=episode_manager,
        ...     reset_manager=reset_manager,
        ...     action_bound=(-10.0, 10.0),
        ...     body_state_dim=12,
        ...     topic_config=topic_cfg,
        ...     loop_config=loop_cfg,
        ...     training_config=train_cfg,
        ...     residual_config=residual_cfg,
        ...     actor_lr=3e-4,
        ...     critic_lr=3e-4
        ... )
        >>> 
        >>> # Start loops
        >>> agent.start()
    """
    
    # Type hints
    acting_agent: OffPolicyAgent
    learning_agent: OffPolicyAgent
    replay_buffer: Union[DualReplayBuffer, CognitiveReplayBuffer]
    
    def __init__(
        self,
        # Core parameters 
        obs_dim: int,
        act_dim: int,
        node: Node,
        replay_buffer: Union[DualReplayBuffer, CognitiveReplayBuffer],
        log_path: str,
        vision_processor: VisionProcessor,
        episode_manager: EpisodeManager,
        reset_manager: ResetManager,
        action_bound: Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor]],
        body_state_dim: int = 12,
        device: str = 'cpu',
        topic_config: Optional[TopicConfig] = None,
        loop_config: Optional[LoopConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        residual_config: Optional[ResidualConfig] = None,
        auto_reset_on_fall: bool = True,  # ⭐ Added
        **agent_kwargs  # Agent hyperparameters 
    ):
        """
        Initialize Distributed DDPG with Configuration Classes
        
        Args:
            obs_dim: Observation dimension (body_state + vision_feature)
            act_dim: Action dimension
            node: ROS2 Node instance
            replay_buffer: Replay buffer instance
            device: PyTorch device ('cpu' or 'cuda:0')
            logger: Logger instance
            vision_processor: VisionProcessor instance
            episode_manager: EpisodeManager instance
            reset_manager: ResetManager instance
            action_bound: Action bounds (min, max)
            body_state_dim: Body state dimension (default: 12)
            topic_config: Topic configuration (default: TopicConfig())
            loop_config: Loop configuration (default: LoopConfig())
            training_config: Training configuration (default: TrainingConfig())
            residual_config: Residual control configuration (default: ResidualConfig())
            auto_reset_on_fall: Auto reset when fall detected (default: True)
            **agent_kwargs: Additional agent arguments (actor_lr, critic_lr, gamma, tau, etc.)
        
        Example:
            >>> agent = DistributedRunner(
            ...     obs_dim=36,
            ...     act_dim=6,
            ...     node=node,
            ...     replay_buffer=buffer,
            ...     device='cuda:0',
            ...     logger=logger,
            ...     vision_processor=vision_processor,
            ...     episode_manager=episode_manager,
            ...     reset_manager=reset_manager,
            ...     action_bound=(-10.0, 10.0),
            ...     body_state_dim=12,
            ...     # Configurations
            ...     topic_config=TopicConfig(),
            ...     loop_config=LoopConfig(action_frequency=10.0),
            ...     training_config=TrainingConfig(batch_size=512),
            ...     residual_config=ResidualConfig(policy_blending_weight=0.3),
            ...     auto_reset_on_fall=True,
            ...     # Agent hyperparameters
            ...     actor_lr=3e-4,
            ...     critic_lr=3e-4,
            ...     gamma=0.99,
            ...     tau=0.005
            ... )
        """
        # Use default configs if not provided
        self.topic_cfg = topic_config or TopicConfig()
        self.loop_cfg = loop_config or LoopConfig()
        self.train_cfg = training_config or TrainingConfig()
        self.residual_cfg = residual_config or ResidualConfig()
        
        # Initialize logger
        self.logger = Logger(log_dir=log_path, device=device)
        self.node.get_logger().info(f"📊 Logger initialized: {log_path}")
        
        # Call parent constructor
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            node=node,
            replay_buffer=replay_buffer,
            device=device,
            logger=self.logger,
            action_bound=action_bound,
            sync_interval=self.train_cfg.sync_interval,
            **agent_kwargs
        )
        
        # Store components
        self.vision_processor = vision_processor
        self.episode_manager = episode_manager
        self.reset_manager = reset_manager
        self.body_state_dim = body_state_dim
        self.auto_reset_on_fall = auto_reset_on_fall
        
        # Setup residual control
        self.residual_gain = self.residual_cfg.get_gain(act_dim, body_state_dim, device)
        print(f"Residual Gain:\n{self.residual_gain}")
        self.policy_blending_weight = self.residual_cfg.policy_blending_weight
        
        # Tracking error state
        self.tracking_error_ = Vec12Msg(data=[0.0] * self.body_state_dim)
        
        # Fall detection state
        self.last_fall_state: bool = False
        self.fall_count: int = 0
        
        # Statistics
        self.total_actor_loss = 0.0
        self.total_critic_loss = 0.0
        self.transition_count = 0
        self.action_count = 0
        self.action_time = 0.0
        self.learn_time = 0.0
        
        # ========== ROS2 Publishers ==========
        self.action_pub = self.create_publisher(EdgeAction, self.topic_cfg.action, 10)
        self.learned_action_pub = self.create_publisher(EdgeAction, self.topic_cfg.learned_action, 10)
        
        # ========== ROS2 Timers (not started yet) ==========
        self.action_timer = None
        self.learning_timer = None
        
        self.node.get_logger().info(
            f"\n{'='*70}\n"
            f"✅ Distributed Agent initialized\n"
            f"{'='*70}\n"
            f"📡 Topics:\n"
            f"   Transition:      {self.topic_cfg.transition}\n"
            f"   Tracking Error:  {self.topic_cfg.tracking_error}\n"
            f"   Action:          {self.topic_cfg.action}\n"
            f"   Learned Action:  {self.topic_cfg.learned_action}\n"
            f"   Fall Detection:  {self.topic_cfg.fall_detection}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"⏱️  Loops:\n"
            f"   Action Freq:     {self.loop_cfg.action_frequency}Hz\n"
            f"   Learning Freq:   {self.loop_cfg.learning_frequency}Hz\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"🎓 Training:\n"
            f"   Batch Size:      {self.train_cfg.batch_size}\n"
            f"   Save Interval:   {self.train_cfg.save_interval}\n"
            f"   Sync Interval:   {self.train_cfg.sync_interval}\n"
            f"   Log Interval:    {self.train_cfg.log_interval}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"🔄 Reset Configuration:\n"
            f"   Auto reset on fall: {self.auto_reset_on_fall}\n"
            f"{'='*70}"
        )
        
        # Log agent info
        self._log_agent_info()
        
        # self.start()  # Start loops immediately upon initialization

    
    # ==================== Loop Control ====================
    def start(self):
        """
        Start acting and learning loops.
        Creates ROS2 timers for action generation and policy learning.
        
        Example:
            >>> agent.start()
            >>> # Loops are now running
        """
        if self.action_timer is not None or self.learning_timer is not None:
            self.node.get_logger().warn("⚠️ Loops already started")
            return
        
        # Create timers
        self.action_timer = self.node.create_timer(
            1.0 / self.loop_cfg.action_frequency,
            self.action_loop
        )
        
        self.learning_timer = self.node.create_timer(
            1.0 / self.loop_cfg.learning_frequency,
            self.learning_loop
        )
        
        self.node.get_logger().info(
            f"\n{'='*70}\n"
            f"🚀 Loops started\n"
            f"{'='*70}\n"
            f"   Action Loop:   {self.loop_cfg.action_frequency}Hz\n"
            f"   Learning Loop: {self.loop_cfg.learning_frequency}Hz\n"
            f"{'='*70}"
        )
    
    def stop(self):
        """
        Stop acting and learning loops. 
        Destroys ROS2 timers.
        
        Example:
            >>> agent.stop()
            >>> # Loops are now stopped
        """
        if self.action_timer is not None:
            self.action_timer.cancel()
            self.action_timer = None
        
        if self.learning_timer is not None:
            self.learning_timer.cancel()
            self.learning_timer = None
        
        self.node.get_logger().info("🛑 Loops stopped")
    
    # ==================== ROS2 Callbacks ====================
    
    def tracking_error_callback(self, msg: Vec12Msg):
        """Tracking error callback"""
        self.tracking_error_ = msg
        
        # Log tracking error statistics
        tracking_errors = torch.tensor(msg.data, dtype=torch.float32)
        self.logger.writer.add_scalar(
            'TrackingError/mean',
            torch.mean(torch.abs(tracking_errors)),
            self.transition_count
        )
        self.logger.writer.add_scalar(
            'TrackingError/max',
            torch.max(torch.abs(tracking_errors)),
            self.transition_count
        )
        self.logger.writer.add_scalar(
            'TrackingError/std',
            torch.std(tracking_errors),
            self.transition_count
        )
    
    def fall_detection_callback(self, msg: Bool) -> None:
        """
        Fall detection callback with proper state management
        
        Strategy:
            1. Detect rising edge (False → True)
            2. Trigger reset
            3. Clear fall state after reset (assume reset fixes the fall)
            4. Allow re-detection if fall happens again
        """
        is_fall: bool = msg.data
        
        # ⭐ Detect rising edge (False → True transition)
        if is_fall and not self.last_fall_state:
            self.fall_count += 1
            
            self.node.get_logger().warn(
                f"\n{'='*70}\n"
                f"⚠️  FALL DETECTED #{self.fall_count}\n"
                f"{'='*70}"
            )
            
            # Log fall detection
            self.logger.writer.add_scalar('Fall/detected', 1, self.fall_count)
            self.logger.writer.add_scalar('Fall/total_count', self.fall_count, self.transition_count)
            
            if self.auto_reset_on_fall:
                self.node.get_logger().info('🔄 Auto reset triggered by fall detection')
                
                # Trigger reset via ResetManager
                success, message = self.reset_manager.call_reset_service(reason=ResetReason.FALL_DETECTED)
                
                if success:
                    self.node.get_logger().info(
                        f"✅ Fall reset completed: {message}\n"
                        f"   Clearing fall state to allow re-detection"
                    )
                    
                    # ⭐ Clear fall state after successful reset
                    # Assumption: Reset fixes the fall, robot is now stable
                    self.last_fall_state = False
                    
                    # ⭐ Optional: Add a cooldown to prevent immediate re-trigger
                    self.fall_reset_time = time.time()
                    
                else:
                    self.node.get_logger().error(
                        f"❌ Fall reset failed: {message}\n"
                        f"   Keeping fall state for retry"
                    )
                    # ⭐ Keep fall state = True to allow retry on next callback
                    self.last_fall_state = True
            else:
                self.node.get_logger().info('⏸️  Auto reset on fall is disabled')
                # ⭐ Update state normally
                self.last_fall_state = is_fall
        
        # ⭐ Detect falling edge (True → False) - robot recovered
        elif not is_fall and self.last_fall_state:
            self.node.get_logger().info(
                f"✅ Robot recovered from fall state"
            )
            self.last_fall_state = False
        
        # ⭐ No state change, just update
        else:
            self.last_fall_state = is_fall

    
    def transition_callback(self, msg: Transition):
        """
        Transition callback - automatically collect transitions
        
        ✅ Skips during reset (managed by ResetManager)
        """
        # Skip during reset
        if self.reset_manager.is_resetting:
            self.node.get_logger().debug('⏸️  Skipping transition during reset')
            return
        
        # Update episode statistics
        reward_components = {
            'obstacle': msg.reward.obstacle_reward,
            'goal_distance': msg.reward.goal_distance_reward,
            'body_state': msg.reward.body_state_reward
        }
        
        stepwise_total_reward = self.episode_manager.update_step(reward_components)
        
        # Update reward normalizer
        self.update_reward_normalizer(stepwise_total_reward)
        
        # Create observation (body_state + vision_feature)
        observation = torch.cat((
            torch.tensor(
                np.array(self.tracking_error_.data, dtype=np.float32),
                dtype=torch.float32,
                device=self.device
            ),
            self.vision_processor.get_last_features()
        ), dim=0)
        
        next_observation = torch.cat((
            torch.tensor(
                np.array(msg.next_obs, dtype=np.float32),
                dtype=torch.float32,
                device=self.device
            ),
            self.vision_processor.get_current_features()
        ), dim=0)
        
        # Add to buffer
        t = self.replay_buffer.to_transition(
            obs=observation,
            action=msg.action,
            reward=stepwise_total_reward,
            next_obs=next_observation,
            done=msg.done
        )
        
        action_mode = ActionMode(msg.action_mode)
        self.replay_buffer.add_transition(t, action_mode=action_mode)
        self.episode_manager.increment_action_count(action_mode == ActionMode.STUDENT)
        
        self.transition_count += 1
        
        # Log stepwise statistics
        self._log_stepwise_stats(stepwise_total_reward, reward_components)
        
        # Check episode done
        if msg.done or self.episode_manager.is_episode_done():
            self._handle_episode_done()
    
    # ==================== Action Loop ====================
    def compute_output_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute output action with residual control
        
        Returns:
            output_action: Final action to send to robot [act_dim]
            drl_action: Pure DRL action (for logging) [act_dim]
        """
        # Get current observation
        observation = torch.cat((
            torch.tensor(
                np.array(self.tracking_error_.data, dtype=np.float32),
                dtype=torch.float32,
                device=self.device
            ),
            self.vision_processor.curr_vision_feature
        ), dim=0)
        
        # Get policy action from acting agent
        policy_action = self.acting_agent.act(
            obs=observation.unsqueeze(0),
            noise_scale=0.1,
            explore=True
        ).squeeze(0)
        
        # Compute residual action (tracking error compensation)
        tracking_error = torch.tensor(
            self.tracking_error_.data,
            dtype=torch.float32,
            device=self.device
        )
        residual_action = (self.residual_gain @ tracking_error.unsqueeze(-1)).flatten()
        
        # Blend with teacher action (for cognitive buffer)
        if isinstance(self.replay_buffer, CognitiveReplayBuffer):
            blended_action = self.replay_buffer.get_weighted_blended_action()
            drl_action = policy_action + self.policy_blending_weight * blended_action
        else:
            drl_action = policy_action
        
        # Final output action
        output_action = drl_action + residual_action
        
        return output_action, drl_action
    
    def action_loop(self):
        """
        Action generation loop (high frequency, e.g., 10Hz)
        
        ✅ Uses acting agent (fast, non-blocking)
        ✅ Skips during reset
        """
        # Skip during reset
        if self.reset_manager.is_resetting:
            return
        
        action_start = time.time()
        
        # Compute actions
        output_action, drl_action = self.compute_output_action()
        
        # Publish output action (with residual)
        edge_action_msg = EdgeAction()
        edge_action_msg.header.stamp = self.node.get_clock().now().to_msg()
        edge_action_msg.control_action = output_action.detach().cpu().numpy().flatten().astype(np.float64)
        edge_action_msg.action_mode = ActionMode.STUDENT.value
        self.action_pub.publish(edge_action_msg)
        
        # Publish learned action (pure DRL, for logging)
        edge_drl_action_msg = EdgeAction()
        edge_drl_action_msg.header.stamp = self.node.get_clock().now().to_msg()
        edge_drl_action_msg.control_action = drl_action.detach().cpu().numpy().flatten().astype(np.float64)
        edge_drl_action_msg.action_mode = ActionMode.STUDENT.value
        self.learned_action_pub.publish(edge_drl_action_msg)
        
        self.action_count += 1
        self.action_time = time.time() - action_start
        
        # Log action generation time
        self.logger.writer.add_scalar('Perf/action_generation_ms', self.action_time * 1000, self.action_count)
        
        # Warn if action generation is slow
        expected_time = 1.0 / self.loop_cfg.action_frequency
        if self.action_time > expected_time * 1.5:
            self.node.get_logger().warn(
                f"⚠️ Slow action generation: {self.action_time*1000:.2f}ms "
                f"(expected <{expected_time*1000:.2f}ms)"
            )
    
    # ==================== Learning Loop ====================
    
    def learning_loop(self):
        """
        Learning loop (low frequency, e.g., 2Hz)
        
        ✅ Uses learning agent (slow, can be blocked)
        ✅ Skips during reset
        """
        import time
        
        # Skip if resetting
        if self.reset_manager.is_resetting:
            self.node.get_logger().debug('⏸️  Skipping learning during reset')
            return
        
        # Logging buffer sizes
        self.logger.writer.add_scalar('Buffer/student_size', self.replay_buffer.stu_buf_size, self.update_count)
        self.logger.writer.add_scalar('Buffer/teacher_size', self.replay_buffer.tea_buf_size, self.update_count)

        # Check if buffers are ready
        if not self.replay_buffer.stu_buf.is_ready:
            self.node.get_logger().info(
                f"Student Buffer not ready for learning. "
                f"Current size: {self.replay_buffer.stu_buf_size}, "
                f"required prefill size: {self.replay_buffer.stu_buf.prefill_size}."
            )
            return
        
        if not self.replay_buffer.tea_buf.is_ready:
            self.node.get_logger().info(
                f"Teacher Buffer not ready for learning. "
                f"Current size: {self.replay_buffer.tea_buf_size}, "
                f"required prefill size: {self.replay_buffer.tea_buf.prefill_size}."
            )
            return

        learn_start = time.time()
        
        # Update policy (using batch_size from training config)
        actor_loss, critic_loss = self.learning_agent.update(self.train_cfg.batch_size)
        
        # Convert to dictionary
        losses = {
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
            'critic_loss': critic_loss.item() if isinstance(critic_loss, torch.Tensor) else critic_loss,
        }
        
        # Update statistics
        self.update_count += 1
        self.total_actor_loss += losses['actor_loss']
        self.total_critic_loss += losses['critic_loss']
        self.learn_time = time.time() - learn_start
        
        # Check if should synchronize
        if self.should_sync(self.update_count):
            sync_time = self.sync_parameters()
            losses['sync_time'] = sync_time
            
            self.node.get_logger().info(
                f"🔄 [Sync #{self.sync_count}] "
                f"Parameters synchronized in {sync_time*1000:.2f}ms"
            )
        
        # Log metrics
        self._log_update_metrics(losses)
        
        # Save models periodically (using save_interval from training config)
        if self.update_count % self.train_cfg.save_interval == 0:
            self.save_checkpoint()
    
    # ==================== Model Management ====================
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        import os
        save_path = f"{self.train_cfg.model_save_path}_{self.update_count}"
        os.makedirs(save_path, exist_ok=True)
        self.save_models(save_path)
        self.node.get_logger().info(f"💾 Models saved to {save_path}")
    
    def save_final_checkpoint(self):
        """Save final model checkpoint"""
        import os
        save_path = f"{self.train_cfg.model_save_path}_final"
        os.makedirs(save_path, exist_ok=True)
        self.save_models(save_path)
        self.node.get_logger().info(f"💾 Final models saved to {save_path}")
    
    # ==================== Episode Management ====================
    def _handle_episode_done(self):
        """
        Handle episode done event
        
        ✅ Logs episode statistics
        ✅ Triggers reset via ResetManager
        """
        # Log episode statistics
        stats, activation_ratio = self.episode_manager.log_episode_end()
        
        # Determine reset reason
        reset_reason = 'timeout' if self.episode_manager.is_episode_done() else 'episode_done'
        
        self.node.get_logger().info(
            f"\n{'='*70}\n"
            f"📊 Episode {self.episode_manager.episode_count} done (reason: {reset_reason})\n"
            f"{'─'*70}\n"
            f"   Total Reward:     {stats['reward']:.2f}\n"
            f"   Episode Length:   {stats['length']}\n"
            f"   Activation Ratio: {activation_ratio:.2%}\n"
            f"{'='*70}"
        )
        
        # Finish episode for reward normalizer
        self.finish_episode()
        
        # Trigger reset via ResetManager
        self.node.get_logger().info(f'🔄 Triggering auto reset (reason: {reset_reason})')
        success, message = self.reset_manager.call_reset_service(reset_reason)
        
        if success:
            self.node.get_logger().info(f'✅ Episode reset completed: {message}')
        else:
            self.node.get_logger().error(f'❌ Episode reset failed: {message}')
    
    # ==================== Reward Normalization ====================
    
    def update_reward_normalizer(self, reward: float):
        """Update reward normalizer"""
        if (hasattr(self.learning_agent, 'reward_normalizer') and 
            self.learning_agent.reward_normalizer is not None):
            self.learning_agent.reward_normalizer.update(reward)
    
    def finish_episode(self):
        """Finish episode for reward normalizer"""
        if (hasattr(self.learning_agent, 'reward_normalizer') and 
            self.learning_agent.reward_normalizer is not None):
            self.learning_agent.reward_normalizer.finish_episode()
    
    # ==================== Statistics ====================
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent-specific statistics"""
        if self.update_count == 0:
            return {
                'avg_actor_loss': 0.0,
                'avg_critic_loss': 0.0,
                'update_count': 0,
                'transition_count': self.transition_count,
                'action_count': self.action_count,
            }
        
        return {
            'avg_actor_loss': self.total_actor_loss / self.update_count,
            'avg_critic_loss': self.total_critic_loss / self.update_count,
            'update_count': self.update_count,
            'transition_count': self.transition_count,
            'action_count': self.action_count,
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'action_time_ms': self.action_time * 1000,
            'learn_time_ms': self.learn_time * 1000,
            'action_frequency': self.loop_cfg.action_frequency,
            'learning_frequency': self.loop_cfg.learning_frequency,
        }
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get network information"""
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'actor_params': count_parameters(self.learning_agent.actor),
            'critic_params': count_parameters(self.learning_agent.critic),
            'target_actor_params': count_parameters(self.learning_agent.target_actor),
            'target_critic_params': count_parameters(self.learning_agent.target_critic),
            'total_params': (
                count_parameters(self.learning_agent.actor) +
                count_parameters(self.learning_agent.critic) +
                count_parameters(self.learning_agent.target_actor) +
                count_parameters(self.learning_agent.target_critic)
            ),
        }
    
    def get_fall_stats(self) -> Dict[str, Any]:
        """⭐ Get fall detection statistics"""
        return {
            'fall_count': self.fall_count,
            'last_fall_state': self.last_fall_state,
            'auto_reset_on_fall': self.auto_reset_on_fall,
        }
    
    def get_reset_stats(self) -> Dict[str, Any]:
        """Get reset statistics from ResetManager"""
        return self.reset_manager.get_statistics()
    
    # ==================== Logging ====================
    
    def _log_agent_info(self):
        """Log agent initialization information"""
        network_info = self.get_network_info()
        
        self.node.get_logger().info(
            f"\n{'='*70}\n"
            f"🎯 Distributed Agent Configuration\n"
            f"{'='*70}\n"
            f"📐 Dimensions:\n"
            f"   Body State Dim:       {self.body_state_dim}\n"
            f"   Vision Feature Dim:   {self.obs_dim - self.body_state_dim}\n"
            f"   Total Obs Dim:        {self.obs_dim}\n"
            f"   Action Dim:           {self.act_dim}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"🎓 Training:\n"
            f"   Batch Size:           {self.train_cfg.batch_size}\n"
            f"   Actor LR:             {self.agent_kwargs.get('actor_lr', 3e-4)}\n"
            f"   Critic LR:            {self.agent_kwargs.get('critic_lr', 3e-4)}\n"
            f"   Gamma:                {self.agent_kwargs.get('gamma', 0.99)}\n"
            f"   Tau:                  {self.agent_kwargs.get('tau', 0.005)}\n"
            f"   Sync Interval:        {self.train_cfg.sync_interval}\n"
            f"   Save Interval:        {self.train_cfg.save_interval}\n"
            f"   Log Interval:         {self.train_cfg.log_interval}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"🧠 Network Parameters:\n"
            f"   Actor:         {network_info['actor_params']:,}\n"
            f"   Critic:        {network_info['critic_params']:,}\n"
            f"   Target Actor:  {network_info['target_actor_params']:,}\n"
            f"   Target Critic: {network_info['target_critic_params']:,}\n"
            f"   Total:         {network_info['total_params']:,}\n"
            f"{'='*70}"
        )
    
    def _log_stepwise_stats(self, total_reward, reward_components):
        """Log stepwise statistics"""
        step = self.episode_manager.current_episode_length
        
        self.logger.writer.add_scalar('Stepwise/total_reward', total_reward, step)
        self.logger.writer.add_scalar('Stepwise/obstacle_reward', reward_components['obstacle'], step)
        self.logger.writer.add_scalar('Stepwise/goal_distance_reward', reward_components['goal_distance'], step)
        self.logger.writer.add_scalar('Stepwise/body_state_reward', reward_components['body_state'], step)
    
    def _log_update_metrics(self, losses: Dict[str, float]):
        """Log update metrics to TensorBoard"""
        # Log losses
        self.logger.writer.add_scalar('Loss/actor_loss', losses['actor_loss'], self.update_count)
        self.logger.writer.add_scalar('Loss/critic_loss', losses['critic_loss'], self.update_count)
        self.logger.writer.add_scalar('Perf/learn_ms', self.learn_time * 1000, self.update_count)
        
        # Log average statistics (using log_interval from training config)
        if self.update_count % self.train_cfg.log_interval == 0:
            stats = self.get_agent_stats()
            perf_stats = self.get_performance_stats()
            sync_stats = self.get_sync_stats()
            reset_stats = self.get_reset_stats()
            fall_stats = self.get_fall_stats()
            
            self.logger.writer.add_scalar('Loss/avg_actor_loss', stats['avg_actor_loss'], self.update_count)
            self.logger.writer.add_scalar('Loss/avg_critic_loss', stats['avg_critic_loss'], self.update_count)
            
            # ⭐ Log reset and fall statistics
            self.logger.writer.add_scalar('Reset/total_count', reset_stats['total'], self.update_count)
            self.logger.writer.add_scalar('Reset/success_rate', reset_stats['success_rate'], self.update_count)
            self.logger.writer.add_scalar('Fall/total_count', fall_stats['fall_count'], self.update_count)
            
            # Console log
            self.node.get_logger().info(
                f"\n{'='*70}\n"
                f"📊 Training Statistics (Update #{self.update_count})\n"
                f"{'='*70}\n"
                f"📉 Losses:\n"
                f"   Actor Loss:      {losses['actor_loss']:.4f}\n"
                f"   Critic Loss:     {losses['critic_loss']:.4f}\n"
                f"   Avg Actor Loss:  {stats['avg_actor_loss']:.4f}\n"
                f"   Avg Critic Loss: {stats['avg_critic_loss']:.4f}\n"
                f"─────────────────────────────────────────────────────────────────────\n"
                f"⚡ Performance:\n"
                f"   Learn Time:      {perf_stats['learn_time_ms']:.2f}ms\n"
                f"   Action Time:     {perf_stats['action_time_ms']:.2f}ms\n"
                f"   Avg Sync Time:   {sync_stats['avg_sync_time']*1000:.2f}ms\n"
                f"─────────────────────────────────────────────────────────────────────\n"
                f"🔢 Counters:\n"
                f"   Updates:         {stats['update_count']}\n"
                f"   Syncs:           {self.sync_count}\n"
                f"   Transitions:     {stats['transition_count']}\n"
                f"   Actions:         {stats['action_count']}\n"
                f"─────────────────────────────────────────────────────────────────────\n"
                f"🔄 Reset Statistics:\n"
                f"   Total Resets:    {reset_stats['total']}\n"
                f"   Success Rate:    {reset_stats['success_rate']:.1%}\n"
                f"   Episode Done:    {reset_stats['reasons']['episode_done']}\n"
                f"   Fall Detected:   {reset_stats['reasons']['fall_detected']}\n"
                f"   Timeout:         {reset_stats['reasons']['timeout']}\n"
                f"   External:        {reset_stats['reasons']['external']}\n"
                f"─────────────────────────────────────────────────────────────────────\n"
                f"⚠️  Fall Detection:\n"
                f"   Total Falls:     {fall_stats['fall_count']}\n"
                f"   Auto Reset:      {'Enabled' if fall_stats['auto_reset_on_fall'] else 'Disabled'}\n"
                f"─────────────────────────────────────────────────────────────────────\n"
                f"💾 Buffer Status:\n"
                f"   Student Size:    {len(self.replay_buffer.stu_buf)}/{self.replay_buffer.stu_buf.capacity}\n"
                f"   Teacher Size:    {len(self.replay_buffer.tea_buf)}/{self.replay_buffer.tea_buf.capacity}\n"
                f"{'='*70}"
            )
    
    # ==================== Utility Methods ====================
    def compute_q_value(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value for given observation-action pair"""
        with torch.no_grad():
            critic_input = self.acting_agent._critic_input(observation, action)
            return self.acting_agent.critic(critic_input)
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"DistributedAgent(\n"
            f"  obs_dim={self.obs_dim},\n"
            f"  act_dim={self.act_dim},\n"
            f"  body_state_dim={self.body_state_dim},\n"
            f"  vision_feature_dim={self.obs_dim - self.body_state_dim},\n"
            f"  action_freq={self.loop_cfg.action_frequency}Hz,\n"
            f"  learning_freq={self.loop_cfg.learning_frequency}Hz,\n"
            f"  sync_interval={self.train_cfg.sync_interval},\n"
            f"  auto_reset_on_fall={self.auto_reset_on_fall},\n"
            f"  device='{self.device}'\n"
            f")"
        )
