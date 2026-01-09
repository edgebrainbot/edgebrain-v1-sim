#!/usr/bin/env python3
"""
Distributed Agent Implementation with Observation & Reward Normalization
"""

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
from edge_drl_student.agents.distributed.ddpg_actor_learner import DDPGActorLearner
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.memory.buffers.off_policy.cognitive_replay_buffer import CognitiveReplayBuffer
from edge_drl_student.processor.vision_processor import VisionProcessor
from edge_drl_student.common.normalizer import EmpiricalNormalizer  # ⭐ Import normalizer


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
            return torch.eye(act_dim, body_state_dim, device=device)
        return self.residual_gain.to(device)


@dataclass
class NormalizationConfig:
    """⭐ Normalization Configuration"""
    normalize_obs: bool = True
    normalize_reward: bool = True
    reward_clip_range: float = 10.0  # Clip normalized reward to [-clip, +clip]
    obs_clip_range: float = 10.0     # Clip normalized obs to [-clip, +clip]


# ==================== Distributed Runner ====================

class DistributedRunner:
    """
    Distributed DDPG Training System with Normalization
    
    Features:
        ✅ Observation normalization (online mean/std)
        ✅ Reward normalization (running mean/std per episode)
        ✅ Composition over inheritance
        ✅ Clear separation of concerns
    
    Example:
        >>> norm_cfg = NormalizationConfig(
        ...     normalize_obs=True,
        ...     normalize_reward=True,
        ...     reward_clip_range=10.0
        ... )
        >>> runner = DistributedRunner(
        ...     obs_dim=36,
        ...     act_dim=6,
        ...     normalization_config=norm_cfg,
        ...     ...
        ... )
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        body_state_dim: int,
        replay_buffer: Union[DualReplayBuffer, CognitiveReplayBuffer],
        action_bound: Union[Tuple[float, float], Tuple[torch.Tensor, torch.Tensor]],
        node: Node,
        logger: Logger,
        device: str = 'cpu',
        
        # Configuration classes
        topic_config: Optional[TopicConfig] = None,
        loop_config: Optional[LoopConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        residual_config: Optional[ResidualConfig] = None,
        normalization_config: Optional[NormalizationConfig] = None,  # ⭐ New
        
        # Components
        vision_processor: Optional[VisionProcessor] = None,
        episode_manager: Optional[EpisodeManager] = None,
        reset_manager: Optional[ResetManager] = None,
        
        # Options
        auto_reset_on_fall: bool = True,
        
        # Agent hyperparameters
        **agent_kwargs
    ):
        """
        Initialize Distributed Runner with Normalization
        
        Args:
            normalization_config: Normalization configuration
            ... (other args same as before)
        """
        
        # Store node and logger
        self.node = node
        self.logger = logger
        
        # Store dimensions
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.body_state_dim = body_state_dim
        self.device = device
        
        # Use default configs if not provided
        self.topic_cfg = topic_config or TopicConfig()
        self.loop_cfg = loop_config or LoopConfig()
        self.train_cfg = training_config or TrainingConfig()
        self.residual_cfg = residual_config or ResidualConfig()
        self.norm_cfg = normalization_config or NormalizationConfig()  # ⭐ New
        
        from edge_drl_student.common.normalizer import RunningMeanStdNormalizer

        # ⭐ Create Normalizers
        if self.norm_cfg.normalize_obs:
            self.obs_normalizer = RunningMeanStdNormalizer(
                shape=(obs_dim,),
                device=device
            )
            self.node.get_logger().info(
                f"✅ Observation normalizer created (RunningMeanStd)"
            )
        else:
            self.obs_normalizer = None
            self.node.get_logger().info("⏸️  Observation normalization disabled")

        if self.norm_cfg.normalize_reward:
            self.reward_normalizer = RunningMeanStdNormalizer(
                shape=(1,),  # 或者 shape=() 如果是标量
                device=device
            )
            self.node.get_logger().info(
                f"✅ Reward normalizer created (RunningMeanStd)"
            )
        else:
            self.reward_normalizer = None
            self.node.get_logger().info("⏸️  Reward normalization disabled")

        
        # ⭐ Create DDPGActorLearner as a component
        self.actor_learner = DDPGActorLearner(
            obs_dim=obs_dim,
            act_dim=act_dim,
            replay_buffer=replay_buffer,
            action_bound=action_bound,
            device=device,
            sync_interval=self.train_cfg.sync_interval,
            **agent_kwargs
        )
        
        # Store components
        self.replay_buffer = replay_buffer
        self.vision_processor = vision_processor
        self.episode_manager = episode_manager
        self.reset_manager = reset_manager
        self.auto_reset_on_fall = auto_reset_on_fall
        
        # Setup residual control
        self.residual_gain = self.residual_cfg.get_gain(act_dim, body_state_dim, device)
        self.node.get_logger().info(f"Residual Gain:\n{self.residual_gain}")
        self.policy_blending_weight = self.residual_cfg.policy_blending_weight
        
        # Tracking error state
        self.tracking_error_ = Vec12Msg(data=[0.0] * self.body_state_dim)
        
        # Fall detection state
        self.last_fall_state: bool = False
        self.fall_count: int = 0
        self.fall_reset_time: float = 0.0
        
        # Statistics
        self.total_actor_loss = 0.0
        self.total_critic_loss = 0.0
        self.transition_count = 0
        self.action_count = 0
        self.action_time = 0.0
        self.learn_time = 0.0
        
        # ⭐ Episode reward accumulator (for reward normalization)
        self.episode_rewards = []
        
        # ========== ROS2 Publishers ==========
        self.action_pub = self.node.create_publisher(
            EdgeAction, 
            self.topic_cfg.action, 
            10
        )
        self.learned_action_pub = self.node.create_publisher(
            EdgeAction, 
            self.topic_cfg.learned_action, 
            10
        )
        
        # ========== ROS2 Subscribers ==========
        self.transition_sub = self.node.create_subscription(
            Transition,
            self.topic_cfg.transition,
            self.transition_callback,
            10
        )
        
        self.tracking_error_sub = self.node.create_subscription(
            Vec12Msg,
            self.topic_cfg.tracking_error,
            self.tracking_error_callback,
            10
        )
        
        self.fall_detection_sub = self.node.create_subscription(
            Bool,
            self.topic_cfg.fall_detection,
            self.fall_detection_callback,
            10
        )
        
        # ========== ROS2 Timers (not started yet) ==========
        self.action_timer = None
        self.learning_timer = None
        
        # Log initialization
        self._log_initialization()
    
    # ==================== Loop Control ====================
    
    def start(self):
        """Start acting and learning loops"""
        if self.action_timer is not None or self.learning_timer is not None:
            print("⚠️ Loops already started")
            return
        
        self.action_timer = self.node.create_timer(
            1.0 / self.loop_cfg.action_frequency,
            self.action_loop
        )
        
        self.learning_timer = self.node.create_timer(
            1.0 / self.loop_cfg.learning_frequency,
            self.learning_loop
        )
        
        print(
            f"\n{'='*70}\n"
            f"🚀 Loops started\n"
            f"{'='*70}\n"
            f"   Action Loop:   {self.loop_cfg.action_frequency}Hz\n"
            f"   Learning Loop: {self.loop_cfg.learning_frequency}Hz\n"
            f"{'='*70}"
        )
    
    def stop(self):
        """Stop acting and learning loops"""
        if self.action_timer is not None:
            self.action_timer.cancel()
            self.action_timer = None
        
        if self.learning_timer is not None:
            self.learning_timer.cancel()
            self.learning_timer = None
        
        print("🛑 Loops stopped")
    
    # ==================== ROS2 Callbacks ====================
    
    def tracking_error_callback(self, msg: Vec12Msg):
        """Tracking error callback"""
        self.tracking_error_ = msg
        
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
        """Fall detection callback with rising edge detection"""
        is_fall: bool = msg.data
        
        # Detect rising edge (False → True)
        if is_fall and not self.last_fall_state:
            self.fall_count += 1
            
            print(
                f"\n{'='*70}\n"
                f"⚠️  FALL DETECTED #{self.fall_count}\n"
                f"{'='*70}"
            )
            
            self.logger.writer.add_scalar('Fall/total_count', self.fall_count, self.transition_count)
            
            if self.auto_reset_on_fall:
                print('🔄 Auto reset triggered by fall detection')
                
                # ⭐ Handle episode done with FALL_DETECTED reason
                self._handle_episode_done(reason=ResetReason.FALL_DETECTED)
                
                # Update fall state
                self.last_fall_state = False
                self.fall_reset_time = time.time()
            else:
                print('⏸️  Auto reset on fall is disabled')
                self.last_fall_state = is_fall
        
        elif not is_fall and self.last_fall_state:
            print("✅ Robot recovered from fall state")
            self.last_fall_state = False
        
        else:
            self.last_fall_state = is_fall

    
    # ==================== Transition Callback ====================
    def transition_callback(self, msg: Transition):
        """Transition callback - collect experiences with normalization"""
        if self.reset_manager.is_resetting:
            print('⏸️  Skipping transition during reset')
            return
        
        # Update episode statistics
        reward_components = {
            'obstacle': msg.reward.obstacle_reward,
            'goal_distance': msg.reward.goal_distance_reward,
            'body_state': msg.reward.body_state_reward
        }
        
        # ⭐ Update episode with new step and get total reward
        stepwise_total_reward = sum(reward_components.values())
        
        # ⭐ Accumulate episode rewards (for reward normalization)
        self.episode_rewards.append(stepwise_total_reward)
        
        # Create observation (body_state + vision_feature)
        observation = torch.cat((
            torch.tensor(
                np.array(msg.obs, dtype=np.float32),
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
        
        # ⭐ Normalize observations
        if self.obs_normalizer is not None:
            
            # Update normalizer statistics
            self.obs_normalizer.update(observation)
            self.obs_normalizer.update(next_observation)
            
            # Normalize observations
            obs_norm = self.obs_normalizer.normalize(observation)
            next_obs_norm = self.obs_normalizer.normalize(next_observation)
            
        else:
            obs_norm = observation
            next_obs_norm = next_observation
        
        # Normalize reward (will be done at episode end)
        
        # Add to buffer
        t = self.replay_buffer.to_transition(
            obs=obs_norm,
            action=msg.action,
            reward=stepwise_total_reward,
            next_obs=next_obs_norm,
            done=msg.done
        )
        
        action_mode = ActionMode(msg.action_mode)

        # Add transition to the self-learning buffer
        if action_mode == ActionMode.STUDENT:
            self.replay_buffer.add_transition(t, action_mode=ActionMode.STUDENT)
            if not self.replay_buffer.stu_buf.is_ready:
                print(f"[Transition] Added transition to student buffer. Current size: {self.replay_buffer.stu_buf_size}")
            
        # Add transition to the teaching-to-learn buffer
        elif action_mode == ActionMode.TEACHER:
            self.replay_buffer.add_transition(t, action_mode=ActionMode.TEACHER)
            if not self.replay_buffer.tea_buf.is_ready:
                print(f"[Transition] Added transition to teacher buffer. Current size: {self.replay_buffer.tea_buf_size}")
        
        else:
            self.node.get_logger().error(f'Unknown action mode: {msg.action_mode}')
            return
        
        self.transition_count += 1
        self.episode_manager.update(reward_components, action_mode)
        
        # Log stepwise statistics
        self._log_stepwise_stats(reward_components)
        
        # Log timestep statistics
        self._log_timestep_stats(reward_components)
        
        # Check episode done
        if msg.done or self.episode_manager.is_episode_done():
            reason = (
                ResetReason.TIMEOUT if self.episode_manager.is_episode_done()
                else ResetReason.EPISODE_DONE
            )
            self._handle_episode_done(reason=reason)

    
    # ==================== Action Loop ====================
    def compute_output_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute output action with residual control and observation normalization
        
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
        
        # Normalize observation
        if self.obs_normalizer is not None:
            obs_norm = self.obs_normalizer.normalize(observation)
        else:
            obs_norm = observation
        
        # Get policy action from actor_learner's acting_agent (using normalized obs)
        policy_action = self.actor_learner.acting_agent.act(
            obs=obs_norm.unsqueeze(0),
            noise_scale=0.1,
            explore=True
        ).squeeze(0)
        
        # Compute residual action
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
        """Action generation loop (high frequency)"""
        if self.reset_manager.is_resetting:
            return
        
        action_start = time.time()
        
        # Compute actions
        output_action, drl_action = self.compute_output_action()
        
        # Publish output action
        edge_action_msg = EdgeAction()
        edge_action_msg.header.stamp = self.node.get_clock().now().to_msg()
        edge_action_msg.control_action = output_action.detach().cpu().numpy().flatten().astype(np.float64)
        edge_action_msg.action_mode = ActionMode.STUDENT.value
        self.action_pub.publish(edge_action_msg)
        
        # Publish learned action
        edge_drl_action_msg = EdgeAction()
        edge_drl_action_msg.header.stamp = self.node.get_clock().now().to_msg()
        edge_drl_action_msg.control_action = drl_action.detach().cpu().numpy().flatten().astype(np.float64)
        edge_drl_action_msg.action_mode = ActionMode.STUDENT.value
        self.learned_action_pub.publish(edge_drl_action_msg)
        
        self.action_count += 1
        self.action_time = time.time() - action_start
        
        # Log
        self.logger.writer.add_scalar('Perf/action_generation_ms', self.action_time * 1000, self.action_count)
        
        # Warn if slow
        # expected_time = 1.0 / self.loop_cfg.action_frequency
        # if self.action_time > expected_time * 1.5:
        #     print(
        #         f"⚠️ Slow action generation: {self.action_time*1000:.2f}ms "
        #         f"(expected <{expected_time*1000:.2f}ms)"
        #     )
    
    # ==================== Learning Loop ====================
    
    def learning_loop(self):
        """Learning loop (low frequency)"""
        if self.reset_manager.is_resetting:
            print('⏸️  Skipping learning during reset')
            return
        
        # Log buffer sizes
        self.logger.writer.add_scalar('Buffer/student_size', self.replay_buffer.stu_buf_size, self.actor_learner.update_count)
        self.logger.writer.add_scalar('Buffer/teacher_size', self.replay_buffer.tea_buf_size, self.actor_learner.update_count)

        # Check if buffers are ready
        if not self.replay_buffer.stu_buf.is_ready:
            print(
                f"Student Buffer not ready. "
                f"Size: {self.replay_buffer.stu_buf_size}/{self.replay_buffer.stu_buf.prefill_size}"
            )
            return
        
        if not self.replay_buffer.tea_buf.is_ready:
            print(
                f"Teacher Buffer not ready. "
                f"Size: {self.replay_buffer.tea_buf_size}/{self.replay_buffer.tea_buf.prefill_size}"
            )
            return

        learn_start = time.time()
        
        # ⭐ Update policy using actor_learner (auto sync)
        losses = self.actor_learner.update(batch_size=self.train_cfg.batch_size)
        
        # Update statistics
        self.total_actor_loss += losses['actor_loss']
        self.total_critic_loss += losses['critic_loss']
        self.learn_time = time.time() - learn_start
        
        # Log sync if it happened
        if 'sync_time' in losses:
            self.node.get_logger().info(
                f"🔄 [Sync #{self.actor_learner.sync_count}] "
                f"Parameters synchronized in {losses['sync_time']*1000:.2f}ms"
            )
        
        # Log metrics
        self._log_update_metrics(losses)
        
        # Save models periodically
        if self.actor_learner.update_count % self.train_cfg.save_interval == 0:
            self.save_checkpoint()
    
    # ==================== Model Management ====================
    
    def save_checkpoint(self):
        """Save model checkpoint including normalizers"""
        import os
        save_path = f"{self.train_cfg.model_save_path}_{self.actor_learner.update_count}"
        os.makedirs(save_path, exist_ok=True)
        
        # Save actor-learner models
        self.actor_learner.save_models(save_path)
        
        # ⭐ Save normalizers
        if self.obs_normalizer is not None:
            torch.save(
                self.obs_normalizer.state_dict(),
                os.path.join(save_path, "obs_normalizer.pt")
            )
        
        if self.reward_normalizer is not None:
            torch.save(
                self.reward_normalizer.state_dict(),
                os.path.join(save_path, "reward_normalizer.pt")
            )
        
        self.node.get_logger().info(f"💾 Models and normalizers saved to {save_path}")
    
    def save_final_checkpoint(self):
        """Save final model checkpoint"""
        import os
        save_path = f"{self.train_cfg.model_save_path}_final"
        os.makedirs(save_path, exist_ok=True)
        
        # Save actor-learner models
        self.actor_learner.save_models(save_path)
        
        # ⭐ Save normalizers
        if self.obs_normalizer is not None:
            torch.save(
                self.obs_normalizer.state_dict(),
                os.path.join(save_path, "obs_normalizer.pt")
            )
        
        if self.reward_normalizer is not None:
            torch.save(
                self.reward_normalizer.state_dict(),
                os.path.join(save_path, "reward_normalizer.pt")
            )
        
        self.node.get_logger().info(f"💾 Final models and normalizers saved to {save_path}")
    
    def load_checkpoint(self, load_path: str):
        """⭐ Load model checkpoint including normalizers"""
        import os
        
        # Load actor-learner models
        self.actor_learner.load_models(load_path)
        
        # Load normalizers
        obs_norm_path = os.path.join(load_path, "obs_normalizer.pt")
        if self.obs_normalizer is not None and os.path.exists(obs_norm_path):
            self.obs_normalizer.load_state_dict(torch.load(obs_norm_path))
            self.node.get_logger().info("✅ Observation normalizer loaded")
        
        reward_norm_path = os.path.join(load_path, "reward_normalizer.pt")
        if self.reward_normalizer is not None and os.path.exists(reward_norm_path):
            self.reward_normalizer.load_state_dict(torch.load(reward_norm_path))
            self.node.get_logger().info("✅ Reward normalizer loaded")
        
        self.node.get_logger().info(f"✅ Models and normalizers loaded from {load_path}")
    
    # ==================== Episode Management ====================
    
    def _handle_episode_done(self, reason: ResetReason = ResetReason.EPISODE_DONE):
        """
        Handle end of episode: log stats, update normalizers, reset environment
        
        Args:
            reason: Reset reason (EPISODE_DONE, FALL_DETECTED, TIMEOUT, etc.)
        """
        
        # ========== Step 1: Log Episode Statistics ==========
        stats, activation_ratio = self.episode_manager.log_episode_end()
        
        # ========== Step 2: Update Reward Normalizer ==========
        if self.reward_normalizer is not None and self.norm_cfg.normalize_reward:
            if len(self.episode_rewards) > 0:
                episode_rewards_tensor = torch.tensor(
                    self.episode_rewards,
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Update normalizer with episode rewards
                for reward in episode_rewards_tensor:
                    self.reward_normalizer.update('reward', reward.unsqueeze(0))
                
                # Log normalization stats
                reward_mean = self.reward_normalizer.get_mean('reward').item()
                reward_std = self.reward_normalizer.get_std('reward').item()
                
                self.logger.writer.add_scalar(
                    'Normalization/reward_mean',
                    reward_mean,
                    self.episode_manager.episode_count
                )
                self.logger.writer.add_scalar(
                    'Normalization/reward_std',
                    reward_std,
                    self.episode_manager.episode_count
                )
                
                self.node.get_logger().info(
                    f"📊 Reward normalization updated: "
                    f"mean={reward_mean:.4f}, std={reward_std:.4f}"
                )
        
        # ========== Step 3: Cleanup Episode State ==========
        self.episode_rewards.clear()
        
        if hasattr(self, 'discounted_return'):
            self.discounted_return = 0.0
        
        # ========== Step 4: Log Episode Summary ==========
        self.node.get_logger().info(
            f"\n{'='*70}\n"
            f"📊 Episode {self.episode_manager.episode_count} ended\n"
            f"{'─'*70}\n"
            f"   Reason:           {reason.value}\n"
            f"   Total Reward:     {stats['reward']:.2f}\n"
            f"   Episode Length:   {stats['length']}\n"
            f"   Activation Ratio: {activation_ratio:.2%}\n"
            f"{'='*70}"
        )
        
        # ========== Step 5: Call Reset Service ==========
        self.node.get_logger().info(f'🔄 Triggering reset (reason: {reason.value})')
        success, message = self.reset_manager.call_reset_service(reason)
        
        if success:
            self.node.get_logger().info(f'✅ Reset completed: {message}')
        else:
            self.node.get_logger().error(f'❌ Reset failed: {message}')

    
    # ==================== Statistics ====================
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        if self.actor_learner.update_count == 0:
            return {
                'avg_actor_loss': 0.0,
                'avg_critic_loss': 0.0,
                'update_count': 0,
                'transition_count': self.transition_count,
                'action_count': self.action_count,
            }
        
        return {
            'avg_actor_loss': self.total_actor_loss / self.actor_learner.update_count,
            'avg_critic_loss': self.total_critic_loss / self.actor_learner.update_count,
            'update_count': self.actor_learner.update_count,
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
            'actor_params': count_parameters(self.actor_learner.learning_agent.actor),
            'critic_params': count_parameters(self.actor_learner.learning_agent.critic),
            'target_actor_params': count_parameters(self.actor_learner.learning_agent.target_actor),
            'target_critic_params': count_parameters(self.actor_learner.learning_agent.target_critic),
            'total_params': (
                count_parameters(self.actor_learner.learning_agent.actor) +
                count_parameters(self.actor_learner.learning_agent.critic) +
                count_parameters(self.actor_learner.learning_agent.target_actor) +
                count_parameters(self.actor_learner.learning_agent.target_critic)
            ),
        }
    
    def get_fall_stats(self) -> Dict[str, Any]:
        """Get fall detection statistics"""
        return {
            'fall_count': self.fall_count,
            'last_fall_state': self.last_fall_state,
            'auto_reset_on_fall': self.auto_reset_on_fall,
        }
    
    def get_reset_stats(self) -> Dict[str, Any]:
        """Get reset statistics"""
        return self.reset_manager.get_statistics()
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """Get sync statistics"""
        return self.actor_learner.get_sync_stats()
    
    def get_normalization_stats(self) -> dict:
        """
        Get current normalization statistics
        
        Returns:
            Dictionary containing normalization statistics
        """
        stats = {}
        
        # ========== Observation Normalizer ==========
        if self.obs_normalizer is not None:
            obs_stats = self.obs_normalizer.get_stats()
            
            # Extract statistics
            stats['obs_mean'] = obs_stats['mean'].mean().item()
            stats['obs_std'] = obs_stats['std'].mean().item()
            stats['obs_count'] = obs_stats['count']
            
            # Optional: Add per-dimension statistics
            stats['obs_mean_per_dim'] = obs_stats['mean'].cpu().numpy().tolist()
            stats['obs_std_per_dim'] = obs_stats['std'].cpu().numpy().tolist()
        
        # ========== Reward Normalizer ==========
        if self.reward_normalizer is not None:
            reward_stats = self.reward_normalizer.get_stats()
            
            # Extract statistics (reward is typically scalar)
            if reward_stats['mean'].numel() == 1:
                stats['reward_mean'] = reward_stats['mean'].item()
                stats['reward_std'] = reward_stats['std'].item()
            else:
                stats['reward_mean'] = reward_stats['mean'].mean().item()
                stats['reward_std'] = reward_stats['std'].mean().item()
            
            stats['reward_count'] = reward_stats['count']
        
        return stats

    
    # ==================== Logging ====================
    
    def _log_initialization(self):
        """Log initialization information"""
        
        self.node.get_logger().info("\n" + "="*70)
        self.node.get_logger().info("📊 DistributedRunner Initialization Summary")
        self.node.get_logger().info("="*70)
        
        # ========== Configuration ==========
        self.node.get_logger().info("\n⚙️  Configuration:")
        self.node.get_logger().info(
            f"   Device:               {self.device}\n"
            f"   Observation Dim:      {self.obs_dim}\n"
            f"   Action Dim:           {self.act_dim}\n"
            f"   Body State Dim:       {self.body_state_dim}"
        )
        
        # ========== Training Config ==========
        self.node.get_logger().info("\n🎓 Training Configuration:")
        self.node.get_logger().info(
            f"   Batch Size:           {self.train_cfg.batch_size}\n"
            # f"   Updates per Step:     {self.train_cfg.updates_per_step}\n"
            f"   Save Interval:        {self.train_cfg.save_interval}\n"
            f"   Log Interval:         {self.train_cfg.log_interval}"
        )
        
        # ========== Actor-Learner Config ==========
        self.node.get_logger().info("\n🔄 Actor-Learner:")
        self.node.get_logger().info(
            f"   Sync Interval:        every {self.actor_learner.sync_interval} updates\n"
            f"   Only Sync Actor:      {self.actor_learner.only_sync_actor}"
        )
        
        # ========== Normalization ==========
        self.node.get_logger().info("\n🔢 Normalization:")
        self.node.get_logger().info(
            f"   Obs Normalization:    {'Enabled' if self.norm_cfg.normalize_obs else 'Disabled'}\n"
            f"   Reward Normalization: {'Enabled' if self.norm_cfg.normalize_reward else 'Disabled'}"
        )
        
        # Get current normalization statistics
        if self.obs_normalizer is not None:
            obs_stats = self.obs_normalizer.get_stats()
            self.node.get_logger().info(
                f"   Obs Stats:            mean={obs_stats['mean'].mean().item():.4f}, "
                f"std={obs_stats['std'].mean().item():.4f}, "
                f"count={obs_stats['count']:.0f}"
            )
        
        if self.reward_normalizer is not None:
            reward_stats = self.reward_normalizer.get_stats()
            # Handle scalar vs vector rewards
            if reward_stats['mean'].numel() == 1:
                mean_val = reward_stats['mean'].item()
                std_val = reward_stats['std'].item()
            else:
                mean_val = reward_stats['mean'].mean().item()
                std_val = reward_stats['std'].mean().item()
            
            self.node.get_logger().info(
                f"   Reward Stats:         mean={mean_val:.4f}, "
                f"std={std_val:.4f}, "
                f"count={reward_stats['count']:.0f}"
            )
        
        # ========== Residual Policy ==========
        # if self.residual_cfg.use_residual:
        #     self.node.get_logger().info("\n🎯 Residual Policy:")
        #     self.node.get_logger().info(
        #         f"   Enabled:              True\n"
        #         f"   Base Policy Type:     {self.residual_cfg.base_policy_type}\n"
        #         f"   Residual Scale:       {self.residual_cfg.residual_scale}"
        #     )
        # else:
        #     self.node.get_logger().info("\n🎯 Residual Policy:      Disabled")
        
        # ========== Agent Info ==========
        self.node.get_logger().info("\n🤖 Agent Configuration:")
        agent_info = self.actor_learner.get_agent_info()
        for key, value in agent_info.items():
            self.node.get_logger().info(f"   {key}: {value}")
        
        # ========== Buffer Info ==========
        self.node.get_logger().info("\n💾 Replay Buffer:")
        # self.node.get_logger().info(
        #     f"   Type:                 {type(self.replay_buffer).__name__}\n"
        #     # f"   Capacity:             {self.replay_buffer.capacity}\n"
        #     f"   Current Size:         {len(self.replay_buffer)}\n"
        #     f"   Prefill Size:         {self.replay_buffer.prefill_size}\n"
        #     f"   Is Ready:             {self.replay_buffer.is_ready()}"
        # )
        
        # ========== Training Status ==========
        self.node.get_logger().info("\n📈 Training Status:")
        self.node.get_logger().info(
            f"   Episode Count:        {self.episode_manager.episode_count}\n"
            # f"   Total Steps:          {self.episode_manager.total_steps}\n"
            f"   Update Count:         {self.actor_learner.update_count}\n"
            f"   Sync Count:           {self.actor_learner.sync_count}"
        )
        
        self.node.get_logger().info("\n" + "="*70 + "\n")


    
    def _log_stepwise_stats(self, reward_components: Dict[str, float]):
        """Log stepwise statistics"""
        step = self.episode_manager.current_episode_length
        stepwise_total_reward = sum(reward_components.values())
        
        self.logger.writer.add_scalar('Stepwise/total_reward', stepwise_total_reward, step)
        self.logger.writer.add_scalar('Stepwise/obstacle_reward', reward_components['obstacle'], step)
        self.logger.writer.add_scalar('Stepwise/goal_distance_reward', reward_components['goal_distance'], step)
        self.logger.writer.add_scalar('Stepwise/body_state_reward', reward_components['body_state'], step)
        self.logger.writer.add_scalar('Stepwise/student_action_count', self.episode_manager.stu_action_count, step)
        self.logger.writer.add_scalar('Stepwise/teacher_action_count', self.episode_manager.tea_action_count, step)
    
    def _log_timestep_stats(self, reward_components: Dict[str, float]):
        """Log timestep statistics"""
        timestep = self.episode_manager.global_steps
        timestep_total_reward = sum(reward_components.values())
        
        self.logger.writer.add_scalar('Timestep/total_reward', timestep_total_reward, timestep)
        self.logger.writer.add_scalar('Timestep/obstacle_reward', reward_components['obstacle'], timestep)
        self.logger.writer.add_scalar('Timestep/goal_distance_reward', reward_components['goal_distance'], timestep)
        self.logger.writer.add_scalar('Timestep/body_state_reward', reward_components['body_state'], timestep)
        self.logger.writer.add_scalar('Timestep/student_action_count', self.episode_manager.stu_action_count, timestep)
        self.logger.writer.add_scalar('Timestep/teacher_action_count', self.episode_manager.tea_action_count, timestep)
    
    def _log_update_metrics(self, losses: Dict[str, float]):
        """Log update metrics"""
        update_count = self.actor_learner.update_count
        
        # Log losses
        self.logger.writer.add_scalar('Loss/actor_loss', losses['actor_loss'], update_count)
        self.logger.writer.add_scalar('Loss/critic_loss', losses['critic_loss'], update_count)
        self.logger.writer.add_scalar('Perf/learn_ms', self.learn_time * 1000, update_count)
        
        # Log average statistics
        if update_count % self.train_cfg.log_interval == 0:
            stats = self.get_agent_stats()
            perf_stats = self.get_performance_stats()
            sync_stats = self.get_sync_stats()
            reset_stats = self.get_reset_stats()
            fall_stats = self.get_fall_stats()
            norm_stats = self.get_normalization_stats()
            
            self.logger.writer.add_scalar('Loss/avg_actor_loss', stats['avg_actor_loss'], update_count)
            self.logger.writer.add_scalar('Loss/avg_critic_loss', stats['avg_critic_loss'], update_count)
            self.logger.writer.add_scalar('Reset/total_count', reset_stats['total'], update_count)
            self.logger.writer.add_scalar('Reset/success_rate', reset_stats['success_rate'], update_count)
            self.logger.writer.add_scalar('Fall/total_count', fall_stats['fall_count'], update_count)
            
            # Console log
            norm_info = ""
            if 'obs_mean' in norm_stats:
                norm_info += f"   Obs Mean/Std:    {norm_stats['obs_mean']:.4f} / {norm_stats['obs_std']:.4f}\n"
            if 'reward_mean' in norm_stats:
                norm_info += f"   Reward Mean/Std: {norm_stats['reward_mean']:.4f} / {norm_stats['reward_std']:.4f}\n"
            
            print(
                f"\n{'='*70}\n"
                f"📊 Training Statistics (Update #{update_count})\n"
                f"{'='*70}\n"
                f"📉 Losses:\n"
                f"   Actor Loss:      {losses['actor_loss']:.4f}\n"
                f"   Critic Loss:     {losses['critic_loss']:.4f}\n"
                f"   Avg Actor Loss:  {stats['avg_actor_loss']:.4f}\n"
                f"   Avg Critic Loss: {stats['avg_critic_loss']:.4f}\n"
                f"─────────────────────────────────────────────────────────────────────\n"
                f"📊 Normalization:\n"
                f"{norm_info}"
                f"─────────────────────────────────────────────────────────────────────\n"
                f"⚡ Performance:\n"
                f"   Learn Time:      {perf_stats['learn_time_ms']:.2f}ms\n"
                f"   Action Time:     {perf_stats['action_time_ms']:.2f}ms\n"
                f"   Avg Sync Time:   {sync_stats['avg_sync_time']*1000:.2f}ms\n"
                f"─────────────────────────────────────────────────────────────────────\n"
                f"🔢 Counters:\n"
                f"   Updates:         {stats['update_count']}\n"
                f"   Syncs:           {self.actor_learner.sync_count}\n"
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
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"DistributedRunner(\n"
            f"  obs_dim={self.obs_dim},\n"
            f"  act_dim={self.act_dim},\n"
            f"  body_state_dim={self.body_state_dim},\n"
            f"  vision_feature_dim={self.obs_dim - self.body_state_dim},\n"
            f"  action_freq={self.loop_cfg.action_frequency}Hz,\n"
            f"  learning_freq={self.loop_cfg.learning_frequency}Hz,\n"
            f"  sync_interval={self.train_cfg.sync_interval},\n"
            f"  normalize_obs={self.norm_cfg.normalize_obs},\n"
            f"  normalize_reward={self.norm_cfg.normalize_reward},\n"
            f"  auto_reset_on_fall={self.auto_reset_on_fall},\n"
            f"  device='{self.device}'\n"
            f")"
        )
