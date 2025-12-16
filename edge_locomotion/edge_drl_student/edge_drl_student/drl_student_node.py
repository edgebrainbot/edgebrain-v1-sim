#!/usr/bin/env python3

import os
import time
import rclpy
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from rclpy.node import Node
from datetime import datetime

from edge_drl_student.common.logger import Logger
from edge_drl_student.utils.utils import ActionMode
from ament_index_python.packages import get_package_prefix
from edge_msgs.msg import Transition, TrackingError, EdgeAction
from edge_drl_student.agents.off_policy.ddpg_agent import DDPGAgent
from edge_drl_student.common.normalizer import RewardNormalizerWindow
from edge_drl_student.memory.buffers.off_policy.solo_replay_buffer import SoloReplayBuffer
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.memory.buffers.off_policy.cognitive_replay_buffer import CognitiveReplayBuffer


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PROJECT_NAME = f"drl_student_cog_{TIMESTAMP}"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "models", PROJECT_NAME)
LOG_PATH = os.path.join(CURRENT_DIR, "logs", PROJECT_NAME)

class DRLStudentNode(Node):
    def __init__(self):
        super().__init__(PROJECT_NAME, namespace='edge')

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.obs_dim = 12
        self.act_dim = 6
        self.policy_blendering_weight = 0.5
        self.save_interval = 300
        self.print_interval = 10
        
        # Initialize Logger
        self.logger = Logger(log_dir=LOG_PATH, device=self.device)
        
        # Initialize Reward Normalizer
        self.reward_normalizer = RewardNormalizerWindow(window_size=1000, clip_range=5.0)
        
        # self.buffer = DualReplayBuffer(student_buf_size=int(1e5), 
        #                                student_prefill_size=int(512),
        #                                teacher_buf_size=int(1e5), 
        #                                teacher_prefill_size=int(50),
        #                                obs_dim=self.obs_dim, 
        #                                action_dim=self.act_dim, 
        #                                device=self.device)
        self.buffer = CognitiveReplayBuffer(student_buf_size=int(1e5), 
                                            student_prefill_size=int(512),
                                            teacher_buf_size=int(1e5), 
                                            teacher_prefill_size=int(50),
                                            obs_dim=self.obs_dim, 
                                            action_dim=self.act_dim, 
                                            device=self.device)
        self.batch_size = 512
        # self.batch_size = 256
        self.last_reload_time = 0.0
        self.tracking_error_ = TrackingError(tracking_error=[0.0] * self.obs_dim)
        
        self.agent = DDPGAgent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            replay_buffer=self.buffer,
            action_bound=(-1.0, 1.0),
            device=self.device
        )

        # ros2 topics
        self.action_pub = self.create_publisher(EdgeAction, 'drl_student_action', 10)
        self.create_subscription(Transition, 'transition', self.transition_callback, 10)
        self.create_subscription(TrackingError, 'tracking_error', self.tracking_error_callback, 10)

        # Create timers
        self.action_timer = self.create_timer(0.1, self.action_loop)  # Generate and publish actions
        self.learn_timer = self.create_timer(1.0, self.learn_step)    # Learning weights updates
        
        # Statistical variables
        self.update_count = 0
        self.action_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.collection_time = 0.0
        self.learn_time = 0.0
        self.iteration_start_time = time.time()

        self.get_logger().info("✅ DRLStudent Node initialized")


    def output_action(self):
        if isinstance(self.buffer, CognitiveReplayBuffer):
            blending_action = self.buffer.get_weighted_blended_action()
            
            return self.policy_action() * (1 - self.policy_blendering_weight) \
                   + blending_action * self.policy_blendering_weight
        else:
            return self.policy_action()

    def policy_action(self):
        return self.agent.act(torch.tensor(
            np.array(self.tracking_error_.tracking_error, dtype=np.float32),
            dtype=torch.float32
        ))      # Non-scaled control action

    def action_loop(self):
        action_start = time.time()
        
        edge_action_msg = EdgeAction()
        edge_action_msg.header.stamp = self.get_clock().now().to_msg()
        edge_action_msg.control_action = self.output_action().detach().cpu().numpy().astype(np.float64) # Scaled control action
        edge_action_msg.action_mode = ActionMode.STUDENT.value
        
        self.action_pub.publish(edge_action_msg)
        
        self.action_count += 1
        self.collection_time = time.time() - action_start

        # self.get_logger().info(f"[Action] Published action (r={msg.action:.3f})")

    def transition_callback(self, msg: Transition):
        # print(f"Received transition: {msg}")
        t = self.buffer.to_transition(
            obs=msg.obs,
            action=msg.action,
            reward=msg.reward,
            next_obs=msg.next_obs,
            done=msg.done
        )
        action_mode = ActionMode(msg.action_mode)

        # Update episode statistics
        self.current_episode_reward += msg.reward
        self.current_episode_length += 1

        # Logging stepwise reward
        self.logger.writer.add_scalar('Step/reward', msg.reward, self.action_count)

        if msg.done:
            self.episode_count += 1
            
            self.logger.add_episode_info(
                reward=self.current_episode_reward,
                length=self.current_episode_length
            )
            
            self.logger.writer.add_scalar('Episode/reward', self.current_episode_reward, self.episode_count)
            self.logger.writer.add_scalar('Episode/length', self.current_episode_length, self.episode_count)
            
            self.get_logger().info(
                f"📊 Episode {self.episode_count} done | "
                f"Reward: {self.current_episode_reward:.2f} | "
                f"Length: {self.current_episode_length}"
            )
            
            # Reset episode statistics
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        # Add transition to the self-learning buffer
        if action_mode == ActionMode.STUDENT:
            self.buffer.add_transition(t, action_mode=ActionMode.STUDENT)
            
            if not self.buffer.stu_buf.is_ready:
                print(f"[Transition] Added transition to student buffer. Current size: {self.buffer.stu_buf_size}")
            
        # Add transition to the teaching-to-learn buffer
        elif action_mode == ActionMode.TEACHER:
            self.buffer.add_transition(t, action_mode=ActionMode.TEACHER)
            
            if not self.buffer.tea_buf.is_ready:
                print(f"[Transition] Added transition to teacher buffer. Current size: {self.buffer.tea_buf_size}")
        
    def tracking_error_callback(self, msg: TrackingError):
        self.tracking_error_ = msg
        
        tracking_errors = np.array(msg.tracking_error)
        self.logger.writer.add_scalar('TrackingError/mean', np.mean(np.abs(tracking_errors)), self.action_count)
        self.logger.writer.add_scalar('TrackingError/max', np.max(np.abs(tracking_errors)), self.action_count)
        self.logger.writer.add_scalar('TrackingError/std', np.std(tracking_errors), self.action_count)
        
        # self.get_logger().info(f"[TrackingError] Received tracking error: {msg}")

    def learn_step(self):
        
        # Logging buffer sizes
        self.logger.writer.add_scalar('Buffer/student_size', self.buffer.stu_buf_size, self.update_count)
        self.logger.writer.add_scalar('Buffer/teacher_size', self.buffer.tea_buf_size, self.update_count)

        if not self.buffer.stu_buf.is_ready:
            self.get_logger().info(f"Student Buffer not ready for learning. Current size: {self.buffer.stu_buf_size}, required prefill size: {self.buffer.stu_buf.prefill_size}.")
            return
        
        if not self.buffer.tea_buf.is_ready:
            self.get_logger().info(f"Teacher Buffer not ready for learning. Current size: {self.buffer.tea_buf_size}, required prefill size: {self.buffer.tea_buf.prefill_size}.")
            return

        learn_start = time.time()
        actor_loss, critic_loss = self.agent.update(self.batch_size)
        self.learn_time = time.time() - learn_start

        # Saving models and logging
        if self.update_count % self.save_interval == 0:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            self.agent.save_models(MODEL_PATH)
            
        # Logging losses
        self.logger.writer.add_scalar('Timing/learn', self.learn_time, self.update_count)
        self.logger.writer.add_scalar('Loss/actor', actor_loss.item(), self.update_count)
        self.logger.writer.add_scalar('Loss/critic', critic_loss.item(), self.update_count)
        self.logger.writer.add_scalar('Loss/total', actor_loss.item() + critic_loss.item(), self.update_count)
        
        self.update_count += 1
        
        if self.update_count % self.print_interval == 0:
            self.get_logger().info(f"Count: {self.update_count}, [Learner] Updated policy (actor_loss={actor_loss.item():.4f}, critic_loss={critic_loss.item():.4f})")


    def normalize(self, reward):
        """归一化奖励"""
        self.current_episode_reward.append(reward)
        
        if len(self.current_episode_reward) < 2:
            return reward
        
        # 计算窗口内的均值和标准差
        mean = np.mean(self.current_episode_reward)
        std = np.std(self.current_episode_reward)
        
        # 标准化
        if std > 1e-8:
            normalized_reward = (reward - mean) / std
        else:
            normalized_reward = reward - mean
        
        # 裁剪
        if self.clip_range is not None:
            normalized_reward = np.clip(
                normalized_reward,
                -self.clip_range,
                self.clip_range
            )
        
        return normalized_reward

# ==========================================================
# 🔹 Main
# ==========================================================
def main(args=None):
    print(f"Model path: {MODEL_PATH}")
    rclpy.init(args=args)
    node = DRLStudentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
