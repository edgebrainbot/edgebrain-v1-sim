#!/usr/bin/env python3

import os
import time
import rclpy
import torch
import numpy as np
from rclpy.node import Node
from datetime import datetime

from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from sensor_msgs.msg import PointCloud2

from edge_drl_student.common.logger import Logger
from edge_drl_student.utils.utils import ActionMode
from edge_msgs.msg import Transition, Vec12Msg, EdgeAction
from edge_drl_student.common.normalizer import RewardNormalizerWindow
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.memory.buffers.off_policy.cognitive_replay_buffer import CognitiveReplayBuffer

# ⭐ Import DDPG Actor-Learner
from edge_drl_student.distributed.ddpg_actor_learner import DDPGActorLearner
from edge_drl_student.core.action_generator import ActionGenerator
from edge_drl_student.common.reset_manager import ResetManager
from edge_drl_student.managers.episode_manager import EpisodeManager
from edge_drl_student.common.vision_processor import VisionProcessor


RUN = 'cognitive'
# RUN = 'dual'

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PROJECT_NAME = f"drl_student_{RUN}_{TIMESTAMP}"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "models", PROJECT_NAME)
LOG_PATH = os.path.join(CURRENT_DIR, "logs/new_data", PROJECT_NAME)

torch.set_printoptions(sci_mode=False)


class DRLStudentNode(Node):
    def __init__(self):
        super().__init__(PROJECT_NAME, namespace='edge')

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Dimensions
        self.body_state_dim = 12
        self.vision_feature_dim = 24
        self.obs_dim = self.body_state_dim + self.vision_feature_dim
        self.act_dim = 6
        
        # Training parameters
        self.batch_size = 512
        self.save_interval = 500
        self.print_interval = 10
        self.max_episode_steps = int(5e4)
        
        # Initialize Logger
        self.logger = Logger(log_dir=LOG_PATH, device=self.device)
        
        # Initialize Reward Normalizer (for episode-level tracking)
        self.reward_normalizer = RewardNormalizerWindow(window_size=1000, clip_range=5.0)
        
        # Initialize tracking error message
        self.tracking_error_ = Vec12Msg(data=[0.0] * self.body_state_dim)
        
        # Initialize replay buffer
        if RUN == 'dual':
            self.buffer = DualReplayBuffer(
                student_buf_size=int(1e5), 
                student_prefill_size=int(self.batch_size),
                teacher_buf_size=int(1e5), 
                teacher_prefill_size=int(50),
                obs_dim=self.obs_dim, 
                action_dim=self.act_dim, 
                device=self.device
            )
        elif RUN == 'cognitive':
            self.buffer = CognitiveReplayBuffer(
                student_buf_size=int(1e5), 
                student_prefill_size=int(self.batch_size),
                teacher_buf_size=int(1e5), 
                teacher_prefill_size=int(50),
                obs_dim=self.obs_dim, 
                action_dim=self.act_dim, 
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported RUN type: {RUN}")
        
        # ========== Initialize Modularized Components ==========
        self.get_logger().info("🔧 Initializing modularized components...")
        
        # ⭐ DDPG Actor-Learner architecture
        # action_bound = (
        #     torch.tensor([-10., -10., -10., -20., -20., -20.], device=self.device),
        #     torch.tensor([ 10.,  10.,  10.,  20.,  20.,  20.], device=self.device)
        # )
        
        self.actor_learner = DDPGActorLearner(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            replay_buffer=self.buffer,
            device=self.device,
            logger=self.logger,
            # action_bound=action_bound,
            sync_interval=5,          # Sync every 5 updates
            log_interval=100,         # ⭐ Log every 100 updates
        )
        
        # Action Generator
        self.action_generator = ActionGenerator(
            body_state_dim=self.body_state_dim,
            device=self.device,
            policy_blending_weight=0.0
        )
        
        # Vision Processor
        self.vision_processor = VisionProcessor(
            device=self.device,
            feature_dim=self.vision_feature_dim
        )
        
        # Episode Manager
        self.episode_manager = EpisodeManager(
            vision_feature_dim=self.vision_feature_dim,
            device=self.device,
            logger=self.logger,
            max_episode_steps=self.max_episode_steps
        )
        
        # Reset Manager
        self.declare_parameter('reset_service_name', '/edge/reset')
        self.declare_parameter('reset_service_timeout', 15.0)
        self.declare_parameter('auto_reset_on_done', True)
        self.declare_parameter('auto_reset_on_fall', True)
        
        self.reset_manager = ResetManager(
            node=self,
            reset_service_name=self.get_parameter('reset_service_name').value,
            reset_timeout=self.get_parameter('reset_service_timeout').value,
            logger=self.logger,
            auto_reset_on_done=self.get_parameter('auto_reset_on_done').value,
            auto_reset_on_fall=self.get_parameter('auto_reset_on_fall').value
        )
        
        self.get_logger().info("✅ All components initialized")

        # ROS2 topics
        self.action_pub = self.create_publisher(EdgeAction, 'drl_student_action', 10)
        self.learned_action_pub = self.create_publisher(EdgeAction, 'drl_student_learned_action', 10)
        self.create_subscription(Transition, 'transition', self.transition_callback, 10)
        self.create_subscription(Vec12Msg, 'tracking_error_stu', self.tracking_error_callback, 10)
        self.create_subscription(PointCloud2, '/registered_scan', self.pointcloud_callback, 10)
        self.create_subscription(Bool, 'is_fall', self.fall_detection_callback, 10)
        
        # Manual Reset Service
        self.manual_reset_service = self.create_service(
            Trigger, 
            'manual_reset',
            self.manual_reset_callback
        )
        
        self.get_logger().info(
            f"✅ Manual reset service created: /edge/manual_reset\n"
            f"   Usage: ros2 service call /edge/manual_reset std_srvs/srv/Trigger"
        )

        # Create timers
        self.action_timer = self.create_timer(0.1, self.action_loop)  # 10Hz
        self.learn_timer = self.create_timer(0.5, self.learning_loop)  # 2Hz
        
        # Statistical variables
        self.update_count = 0
        self.action_count = 0
        self.collection_time = 0.0
        self.learn_time = 0.0
        
        self.get_logger().info("✅ DRLStudent Node initialized")

    # ========== Service Callbacks ==========
    def manual_reset_callback(self, request, response):
        """Manual reset service callback"""
        self.get_logger().info('🔄 Manual reset triggered by service call')
        
        if self.reset_manager.is_resetting:
            response.success = False
            response.message = 'Reset already in progress, please wait'
            return response
        
        success = self.reset_episode(reason='manual')
        
        response.success = success
        if success:
            response.message = (
                f'Manual reset successful. '
                f'Episode #{self.episode_manager.episode_count} started.'
            )
        else:
            response.message = 'Manual reset failed. Check controller service.'
        
        return response

    # ========== ROS2 Callbacks ==========
    def fall_detection_callback(self, msg: Bool):
        """Fall detection callback"""
        should_reset = self.reset_manager.handle_fall_detection(msg.data)
        
        if should_reset:
            self.reset_episode(reason='fall_detected')
    
    def tracking_error_callback(self, msg: Vec12Msg):
        """Tracking error callback"""
        self.tracking_error_ = msg
        
        tracking_errors = np.array(msg.data, dtype=np.float32)
        self.logger.writer.add_scalar('TrackingError/mean', np.mean(np.abs(tracking_errors)), self.action_count)
        self.logger.writer.add_scalar('TrackingError/max', np.max(np.abs(tracking_errors)), self.action_count)
        self.logger.writer.add_scalar('TrackingError/std', np.std(tracking_errors), self.action_count)
    
    def pointcloud_callback(self, msg):
        """Point cloud callback"""
        points = self.vision_processor.parse_pointcloud(msg)
        
        if points is not None:
            self.episode_manager.curr_vision_feature = self.vision_processor.process_pointcloud(points)

    def transition_callback(self, msg: Transition):
        """Transition callback"""
        # Ignore transitions during reset
        if self.reset_manager.is_resetting:
            return
        
        # Update episode statistics
        reward_components = {
            'obstacle': msg.reward.obstacle_reward,
            'goal_distance': msg.reward.goal_distance_reward,
            'body_state': msg.reward.body_state_reward
        }
        
        stepwise_total_reward = self.episode_manager.update_step(reward_components)
        
        # ⭐ Update reward normalizer in actor_learner
        self.actor_learner.update_reward_normalizer(stepwise_total_reward)
        
        # Create observation
        observation = torch.cat((
            torch.tensor(np.array(self.tracking_error_.data, dtype=np.float32), 
                        dtype=torch.float32, device=self.device),
            self.episode_manager.last_vision_feature
        ), dim=0)
        
        next_observation = torch.cat((
            torch.tensor(np.array(msg.next_obs, dtype=np.float32), 
                        dtype=torch.float32, device=self.device),
            self.episode_manager.curr_vision_feature
        ), dim=0)
        
        # Add to buffer
        t = self.buffer.to_transition(
            obs=observation,
            action=msg.action,
            reward=stepwise_total_reward,
            next_obs=next_observation,
            done=msg.done
        )
        
        action_mode = ActionMode(msg.action_mode)
        self.buffer.add_transition(t, action_mode=action_mode)
        self.episode_manager.increment_action_count(action_mode == ActionMode.STUDENT)
        
        # Log stepwise statistics
        self._log_stepwise_stats(stepwise_total_reward, reward_components)
        
        # Check episode done
        if msg.done or self.episode_manager.is_episode_done():
            self._handle_episode_done()

    # ========== Main Loops ==========
    def action_loop(self):
        """Action generation loop (10Hz)"""
        action_start = time.time()
        
        # Get tracking error
        tracking_error = torch.tensor(
            np.array(self.tracking_error_.data, dtype=np.float32),
            dtype=torch.float32,
            device=self.device
        )
        
        # Generate action
        output_action, drl_action = self.action_generator.generate_action(
            actor_learner=self.actor_learner,
            buffer=self.buffer,
            tracking_error=tracking_error,
            vision_feature=self.episode_manager.curr_vision_feature
        )
        
        # Publish actions
        edge_action_msg = EdgeAction()
        edge_action_msg.header.stamp = self.get_clock().now().to_msg()
        edge_action_msg.control_action = output_action.detach().cpu().numpy().flatten().astype(np.float64)
        edge_action_msg.action_mode = ActionMode.STUDENT.value
        
        edge_drl_action_msg = EdgeAction()
        edge_drl_action_msg.header.stamp = self.get_clock().now().to_msg()
        edge_drl_action_msg.control_action = drl_action.detach().cpu().numpy().flatten().astype(np.float64)
        edge_drl_action_msg.action_mode = ActionMode.STUDENT.value
        
        self.action_pub.publish(edge_action_msg)
        self.learned_action_pub.publish(edge_drl_action_msg)
        
        self.action_count += 1
        self.collection_time = time.time() - action_start
        
        self.logger.writer.add_scalar('Timing/action_generation_ms', 
                                      self.collection_time * 1000, 
                                      self.action_count)

    def learning_loop(self):
        """Learning loop (2Hz)"""
        if self.reset_manager.is_resetting:
            return
        
        # Check buffer readiness
        if not self.buffer.stu_buf.is_ready or not self.buffer.tea_buf.is_ready:
            return

        learn_start = time.time()
        
        # ⭐ Update policy (returns dict now)
        losses = self.actor_learner.update(self.batch_size)
        
        self.learn_time = time.time() - learn_start
        self.update_count += 1
        
        # ⭐ Sync info is included in losses dict if synced
        if 'sync_time' in losses:
            self.get_logger().info(
                f"🔄 [Sync #{self.actor_learner.sync_count}] "
                f"Parameters synchronized in {losses['sync_time']*1000:.2f}ms"
            )
        
        # Save models
        if self.update_count % self.save_interval == 0:
            save_path = f"{MODEL_PATH}_{self.update_count}"
            os.makedirs(save_path, exist_ok=True)
            self.actor_learner.training_agent.save_models(save_path)
            self.get_logger().info(f"💾 Models saved to {save_path}")
        
        # ⭐ Log losses (already logged in actor_learner, but we can add extra logs here)
        self.logger.writer.add_scalar('Timing/learn_ms', self.learn_time * 1000, self.update_count)
        
        # ⭐ Print interval logging (actor_learner already logs every log_interval)
        if self.update_count % self.print_interval == 0:
            stats = self.actor_learner.get_ddpg_stats()
            self.get_logger().info(
                f"[Update {self.update_count}] "
                f"actor_loss={losses['actor_loss']:.4f}, "
                f"critic_loss={losses['critic_loss']:.4f}, "
                f"avg_actor_loss={stats['avg_actor_loss']:.4f}, "
                f"avg_critic_loss={stats['avg_critic_loss']:.4f}, "
                f"time={self.learn_time*1000:.2f}ms"
            )

    # ========== Helper Methods ==========
    def reset_episode(self, reason='unknown'):
        """Reset episode"""
        success, message = self.reset_manager.call_reset_service(reason)
        
        if success:
            self.episode_manager.reset_episode_stats()
            self.reset_manager.last_fall_state = False
            
            # ⭐ Finish episode for reward normalizer
            self.actor_learner.finish_episode()
            
            stats = self.reset_manager.get_statistics()
            self.get_logger().info(
                f'✅ Episode reset complete (reason: {reason}). '
                f'Total resets: {stats["total"]} '
                f'(Success: {stats["success"]}, Failed: {stats["failure"]})'
            )
            
            self.get_logger().info(
                f'📊 Reset statistics: '
                f'episode_done={stats["reasons"]["episode_done"]}, '
                f'fall_detected={stats["reasons"]["fall_detected"]}, '
                f'manual={stats["reasons"]["manual"]}, '
                f'timeout={stats["reasons"]["timeout"]}'
            )
        else:
            self.get_logger().error(f'❌ Episode reset failed (reason: {reason}): {message}')
        
        return success
    
    def _handle_episode_done(self):
        """Handle episode done"""
        stats, activation_ratio = self.episode_manager.log_episode_end()
        
        reset_reason = 'timeout' if self.episode_manager.is_episode_done() else 'episode_done'
        
        self.get_logger().info(
            f"📊 Episode {self.episode_manager.episode_count} done (reason: {reset_reason}) | "
            f"Total Reward: {stats['reward']:.2f} | "
            f"Length: {stats['length']} | "
            f"Activation Ratio: {activation_ratio:.2f}"
        )
        
        if self.reset_manager.auto_reset_on_done:
            self.reset_episode(reason=reset_reason)
    
    def _log_stepwise_stats(self, total_reward, reward_components):
        """Log stepwise statistics"""
        step = self.episode_manager.current_episode_length
        
        self.logger.writer.add_scalar('Stepwise/total_reward', total_reward, step)
        self.logger.writer.add_scalar('Stepwise/obstacle_reward', reward_components['obstacle'], step)
        self.logger.writer.add_scalar('Stepwise/goal_distance_reward', reward_components['goal_distance'], step)
        self.logger.writer.add_scalar('Stepwise/body_state_reward', reward_components['body_state'], step)


def main(args=None):
    print(f"Model path: {MODEL_PATH}")
    rclpy.init(args=args)
    node = DRLStudentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
