#!/usr/bin/env python3

import os
import time
import rclpy
import random
import std_msgs.msg
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from rclpy.node import Node
from datetime import datetime

from std_srvs.srv import Trigger
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from edge_drl_student.common.logger import Logger
from edge_drl_student.utils.utils import ActionMode
from ament_index_python.packages import get_package_prefix
from edge_msgs.msg import Transition, TrackingError, EdgeAction
from edge_drl_student.agents.off_policy.ddpg_agent import DDPGAgent
from edge_drl_student.common.normalizer import RewardNormalizerWindow
from edge_drl_student.memory.buffers.off_policy.solo_replay_buffer import SoloReplayBuffer
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.memory.buffers.off_policy.cognitive_replay_buffer import CognitiveReplayBuffer


RUN = 'cognitive'
# RUN = 'dual'

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PROJECT_NAME = f"drl_student_{RUN}_{TIMESTAMP}"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "models", PROJECT_NAME)
LOG_PATH = os.path.join(CURRENT_DIR, "logs/new_data", PROJECT_NAME)

# Set torch print options for better readability
torch.set_printoptions(sci_mode=False)

class DRLStudentNode(Node):
    def __init__(self):
        super().__init__(PROJECT_NAME, namespace='edge')

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.RESIDUAL_GAIN = torch.tensor([
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 35.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=torch.float32, device=self.device) * -0.1


        self.body_state_dim = 12  # Example dimension for body state
        self.vision_feature_dim = 24  # Example dimension for vision features from lidar
        self.obs_dim = self.body_state_dim + self.vision_feature_dim  # 12 from tracking error + 24 from lidar features
        
        self.act_dim = 6
        self.policy_blendering_weight = 0.0
        self.save_interval = 300
        self.print_interval = 10
        self.max_episode_steps = 15000
        
        # Initialize Logger
        self.logger = Logger(log_dir=LOG_PATH, device=self.device)
        
        # Initialize Reward Normalizer
        self.reward_normalizer = RewardNormalizerWindow(window_size=1000, clip_range=5.0)
        
        self.batch_size = 512
        # self.batch_size = 256
        self.last_reload_time = 0.0
        self.tracking_error_ = TrackingError(tracking_error=[0.0] * self.body_state_dim)
        self.curr_vision_feature = torch.zeros(self.vision_feature_dim, device=self.device)
        self.last_vision_feature = torch.zeros(self.vision_feature_dim, device=self.device)
        
        if RUN == 'dual':
            self.buffer = DualReplayBuffer(student_buf_size=int(1e5), 
                                           student_prefill_size=int(self.batch_size),
                                           teacher_buf_size=int(1e5), 
                                           teacher_prefill_size=int(50),
                                           obs_dim=self.obs_dim, 
                                           action_dim=self.act_dim, 
                                           device=self.device)
        elif RUN == 'cognitive':
            print(f"obs_dim: {self.obs_dim}, act_dim: {self.act_dim}")
            self.buffer = CognitiveReplayBuffer(student_buf_size=int(1e5), 
                                                student_prefill_size=int(self.batch_size),
                                                teacher_buf_size=int(1e5), 
                                                teacher_prefill_size=int(50),
                                                obs_dim=self.obs_dim, 
                                                action_dim=self.act_dim, 
                                                device=self.device)
        else:
            raise ValueError(f"Unsupported RUN type: {RUN}")
        
        self.agent = DDPGAgent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            replay_buffer=self.buffer,
            action_bound=(-10.0, 10.0),
            device=self.device
        )

        # ros2 topics
        self.action_pub = self.create_publisher(EdgeAction, 'drl_student_action', 10)
        self.learned_action_pub = self.create_publisher(EdgeAction, 'drl_student_learned_action', 10)
        self.create_subscription(Transition, 'transition', self.transition_callback, 10)
        self.create_subscription(TrackingError, 'tracking_error_stu', self.tracking_error_callback, 10)
        self.create_subscription(PointCloud2, '/registered_scan', self.pointcloud_callback, 10)
        
        # ========== Reset Service Client ==========
        self.declare_parameter('reset_service_name', '/edge/reset')
        self.declare_parameter('reset_service_timeout', 5.0)
        self.declare_parameter('auto_reset_on_done', True)  # Whether to auto reset when episode done
        
        self.reset_service_name = self.get_parameter('reset_service_name').value
        self.reset_service_timeout = self.get_parameter('reset_service_timeout').value
        self.auto_reset_on_done = self.get_parameter('auto_reset_on_done').value
        
        self.reset_client = self.create_client(Trigger, self.reset_service_name)
        self.reset_count = 0
        self.reset_success_count = 0
        self.reset_failure_count = 0
        
        self.get_logger().info(f"Reset service client created: {self.reset_service_name}")
        self.get_logger().info(f"Auto reset on episode done: {self.auto_reset_on_done}")
    

        self.obstacle_reward = 0.0
        self.goal_distance_reward = 0.0
        self.body_state_reward = 0.0

        # Create timers
        self.action_timer = self.create_timer(0.1, self.action_loop)  # Generate and publish actions
        self.learn_timer = self.create_timer(0.5, self.learn_step)    # Learning weights updates
        
        # Statistical variables
        self.update_count = 0
        self.action_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.collection_time = 0.0
        self.learn_time = 0.0
        self.iteration_start_time = time.time()
        self.episode_count = 0
        self.stu_action_count = 0
        self.tea_action_count = 0
        
        # Matrix processing lidar
        self.points = None
        self.mat_base_vec = torch.arange(0.1, 144.1, 0.1, dtype=torch.float32).to(self.device) * 50
        # x = torch.log(self.mat_base_vec) * 0.1        # (original version)
        x = torch.log(self.mat_base_vec) * 0.0001
        
        self.vis_feature_mat = torch.vander(x, N=8, increasing=True).T.to(self.device)
        self.stu_tracking_error = torch.zeros(self.body_state_dim, device=self.device)
        
        self.get_logger().info("✅ DRLStudent Node initialized")

    # ========== Reset Service Methods ==========
    def call_reset_service(self):
        """
        Call reset service to reset controller
        
        Returns:
            tuple: (success: bool, message: str)
        """
        self.reset_count += 1
        
        # Wait for service
        self.get_logger().info(f'[Reset #{self.reset_count}] Waiting for service: {self.reset_service_name}...')
        
        if not self.reset_client.wait_for_service(timeout_sec=self.reset_service_timeout):
            error_msg = f'Service {self.reset_service_name} not available after {self.reset_service_timeout}s'
            self.get_logger().error(error_msg)
            self.reset_failure_count += 1
            return False, error_msg
        
        # Create request
        request = Trigger.Request()
        
        # Call service
        self.get_logger().info('Calling reset service...')
        reset_start = time.time()
        
        try:
            future = self.reset_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=self.reset_service_timeout)
            
            if future.result() is not None:
                response = future.result()
                elapsed_time = time.time() - reset_start
                
                if response.success:
                    self.reset_success_count += 1
                    self.get_logger().info(
                        f'✓ Reset successful in {elapsed_time:.2f}s: {response.message}'
                    )
                    
                    # Log reset statistics
                    self.logger.writer.add_scalar('Reset/success', 1, self.reset_count)
                    self.logger.writer.add_scalar('Reset/time', elapsed_time, self.reset_count)
                    
                else:
                    self.reset_failure_count += 1
                    self.get_logger().warn(
                        f'✗ Reset failed in {elapsed_time:.2f}s: {response.message}'
                    )
                    self.logger.writer.add_scalar('Reset/success', 0, self.reset_count)
                
                return response.success, response.message
            else:
                error_msg = 'Service call failed - no response received'
                self.get_logger().error(error_msg)
                self.reset_failure_count += 1
                self.logger.writer.add_scalar('Reset/success', 0, self.reset_count)
                return False, error_msg
                
        except Exception as e:
            error_msg = f'Exception during service call: {str(e)}'
            self.get_logger().error(error_msg)
            self.reset_failure_count += 1
            self.logger.writer.add_scalar('Reset/success', 0, self.reset_count)
            return False, error_msg
    
    def reset_episode(self):
        """
        Reset episode (call reset service and reset internal states)
        """
        # Call reset service
        success, message = self.call_reset_service()
        
        if success:
            # Reset internal states
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
            self.obstacle_reward = 0.0
            self.goal_distance_reward = 0.0
            self.body_state_reward = 0.0
            self.stu_action_count = 0
            self.tea_action_count = 0
            
            # Reset vision features
            self.curr_vision_feature = torch.zeros(self.vision_feature_dim, device=self.device)
            self.last_vision_feature = torch.zeros(self.vision_feature_dim, device=self.device)
            
            self.get_logger().info(f'Episode reset complete. Total resets: {self.reset_count}')
            
            # Log reset statistics
            success_rate = self.reset_success_count / self.reset_count if self.reset_count > 0 else 0.0
            self.logger.writer.add_scalar('Reset/total_count', self.reset_count, self.episode_count)
            self.logger.writer.add_scalar('Reset/success_rate', success_rate, self.episode_count)
            
        else:
            self.get_logger().error(f'Episode reset failed: {message}')
        
        return success

    def output_action(self):
        
        residual_action = self.RESIDUAL_GAIN @ self.stu_tracking_error.unsqueeze(-1)
        residual_action = residual_action.flatten() 
        print(f"residual_action: {residual_action}")
        
        if isinstance(self.buffer, CognitiveReplayBuffer):
            blending_action = self.buffer.get_weighted_blended_action()
            drl_action = self.policy_action() * 1.0 + blending_action * self.policy_blendering_weight
        else:
            drl_action = self.policy_action() 
            
        output_action = drl_action * 1.0 + residual_action   # Add residual action
        # output_action = drl_action   # Add residual action
        
        return output_action, drl_action 

    def policy_action(self):
        observation = torch.cat((
            torch.tensor(
                np.array(self.tracking_error_.tracking_error, dtype=np.float32),
                dtype=torch.float32,
                device=self.device
            ),
            self.curr_vision_feature
        ), dim=0)
        
        return self.agent.act(observation.unsqueeze(0))      # Non-scaled control action

    def action_loop(self):
        action_start = time.time()
        
        # ️ Generate and publish action
        edge_action_msg = EdgeAction()
        edge_action_msg.header.stamp = self.get_clock().now().to_msg()
        edge_action_msg.control_action = self.output_action()[0].detach().cpu().numpy().flatten().astype(np.float64)
        edge_action_msg.action_mode = ActionMode.STUDENT.value
        
        # Learned action message
        edge_drl_action_msg = EdgeAction()
        edge_drl_action_msg.header.stamp = self.get_clock().now().to_msg()
        edge_drl_action_msg.control_action = self.output_action()[1].detach().cpu().numpy().flatten().astype(np.float64)
        edge_drl_action_msg.action_mode = ActionMode.STUDENT.value
        
        self.action_pub.publish(edge_action_msg)                    # Publish action to control the robot
        self.learned_action_pub.publish(edge_drl_action_msg)        # Publish learned action for logging    
        
        self.action_count += 1
        self.collection_time = time.time() - action_start

        self.t = time.time()
        # self.get_logger().info(f"[Action] Published action (r={msg.action:.3f})")

    def pointcloud_callback(self, msg):
        """
        Callback function for processing incoming PointCloud2 messages
        """
        # self.get_logger().info('---------- laserCloudHandler called ---------')
        
        # Parse point cloud
        self.points = self.parse_pointcloud(msg)
        
        t2 = time.time()
        # self.get_logger().info(f'Point cloud parsing time: {t2 - self.t:.4f} seconds')
        self.t = t2
        # Process points as needed
        if self.points is not None:
            # self.get_logger().info(f'Parsed {len(self.points)} points from point cloud.')
            pass
            # for i, point in enumerate(points[:5]):
            #     self.get_logger().info(f'Point {i}: x={point[0]:.3f}, y={point[1]:.3f}, z={point[2]:.3f}')
    
        points = torch.tensor(self.points, dtype=torch.float32).to(self.device)
        # print(f"points shape: {points.shape}")
        # print(f"self.lidar_matrix shape: {self.lidar_matrix.shape}")
        res =  self.vis_feature_mat @ points 
        res *= 0.01
        
        self.curr_vision_feature = res.flatten()
        # print(f"self.curr_vision_feature: {self.curr_vision_feature}")
        # print(f"curr_vision_feature shape: {self.curr_vision_feature.shape}")
        # print(f"res shape: {res.shape}")
        # print(f"res: {res}")
    
    
    def parse_pointcloud(self, msg):
        """
        Parse PointCloud2 message to extract XYZ coordinates
        """
        try:
            # Use point_cloud2 module to read points
            points_list = []
            
            # Read point cloud data (x, y, z, intensity fields)
            # for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False):
                clean_point = np.nan_to_num([point[0], point[1], point[2]], 
                                 nan=0.0, posinf=0.0, neginf=0.0)
                points_list.append(clean_point.tolist())
                # self.get_logger().info(f'Raw point data: x={clean_point[0]}, y={clean_point[1]}, z={clean_point[2]}')

            
            return np.array(points_list)
            
        except Exception as e:
            self.get_logger().error(f'Failed to parse point cloud data: {str(e)}')
            return None


    def transition_callback(self, msg: Transition):
        # ️Process received transition and add to replay buffer
        self.goal_distance_reward = msg.reward.goal_distance_reward
        self.body_state_reward = msg.reward.body_state_reward
        self.obstacle_reward = msg.reward.obstacle_reward
        
        # Compute total reward
        stepwise_total_reward = self.obstacle_reward + self.goal_distance_reward + self.body_state_reward

        self.stu_tracking_error = torch.tensor(msg.obs,
            dtype=torch.float32,
            device=self.device
        )
        
        observation = torch.cat((
            torch.tensor(
                np.array(self.tracking_error_.tracking_error, dtype=np.float32),
                dtype=torch.float32,
                device=self.device
            ),
            self.last_vision_feature
        ), dim=0)
        
        next_observation = torch.cat((
            torch.tensor(
                np.array(msg.next_obs, dtype=np.float32),
                dtype=torch.float32,
                device=self.device
            ),
            self.curr_vision_feature
        ), dim=0)
        
        # print(f"observation in transition_callback: {observation} shape: {observation.shape}")
        # print(f"msg.next_obs in transition_callback: {msg.next_obs} shape: {len(msg.next_obs)}")
        t = self.buffer.to_transition(
            obs=observation,
            action=msg.action,
            reward=stepwise_total_reward,
            next_obs=next_observation,
            done=msg.done
        )
        action_mode = ActionMode(msg.action_mode)

        # Add transition to the self-learning buffer
        if action_mode == ActionMode.STUDENT:
            self.buffer.add_transition(t, action_mode=ActionMode.STUDENT)
            self.stu_action_count += 1
            
            if not self.buffer.stu_buf.is_ready:
                print(f"[Transition] Added transition to student buffer. Current size: {self.buffer.stu_buf_size}")
            
        # Add transition to the teaching-to-learn buffer
        elif action_mode == ActionMode.TEACHER:
            self.buffer.add_transition(t, action_mode=ActionMode.TEACHER)
            self.tea_action_count += 1
            
            if not self.buffer.tea_buf.is_ready:
                print(f"[Transition] Added transition to teacher buffer. Current size: {self.buffer.tea_buf_size}")
        
        # Update episode statistics
        self.current_episode_reward += stepwise_total_reward
        self.current_episode_length += 1
        
        # Logging stepwise statistics
        self.logger.writer.add_scalar('Stepwise/total_reward', stepwise_total_reward, self.current_episode_length)
        self.logger.writer.add_scalar('Stepwise/obstacle_reward', self.obstacle_reward, self.current_episode_length)
        self.logger.writer.add_scalar('Stepwise/goal_distance_reward', self.goal_distance_reward, self.current_episode_length)
        self.logger.writer.add_scalar('Stepwise/body_state_reward', self.body_state_reward, self.current_episode_length)
        self.logger.writer.add_scalar('Stepwise/student_action_count', self.stu_action_count, self.current_episode_length)
        self.logger.writer.add_scalar('Stepwise/teacher_action_count', self.tea_action_count, self.current_episode_length)
        
        # Episode done processing
        if msg.done or self.current_episode_length >= self.max_episode_steps:
            self.episode_count += 1
            
            self.logger.add_episode_info(
                reward=self.current_episode_reward,
                length=self.current_episode_length
            )
            
            activation_ratio = self.stu_action_count / (self.stu_action_count + self.tea_action_count) if (self.stu_action_count + self.tea_action_count) > 0 else 0
            
            self.logger.writer.add_scalar('Episode/length', self.current_episode_length, self.episode_count)
            self.logger.writer.add_scalar('Episode/total_reward', self.current_episode_reward, self.episode_count)
            self.logger.writer.add_scalar('Episode/obstacle_reward', self.obstacle_reward, self.episode_count)
            self.logger.writer.add_scalar('Episode/goal_distance_reward', self.goal_distance_reward, self.episode_count)
            self.logger.writer.add_scalar('Episode/body_state_reward', self.body_state_reward, self.episode_count)
            self.logger.writer.add_scalar('Episode/avg_reward_per_step', self.current_episode_reward / self.current_episode_length, 
                                          self.episode_count)
            self.logger.writer.add_scalar('Episode/student_action_count', self.stu_action_count, self.episode_count)
            self.logger.writer.add_scalar('Episode/teacher_action_count', self.tea_action_count, self.episode_count)
            self.logger.writer.add_scalar('Episode/activation_ratio', activation_ratio, self.episode_count)
            
            self.get_logger().info(
                f"📊 Episode {self.episode_count} done | "
                f"Total Reward: {self.current_episode_reward:.2f} | "
                f"Length: {self.current_episode_length} | "
                f"Obstacle Reward: {self.obstacle_reward:.2f} | "
                f"Goal Distance Reward: {self.goal_distance_reward:.2f} | "
                f"Body State Reward: {self.body_state_reward:.2f} | "
                f"Student Action Count: {self.stu_action_count} | "
                f"Teacher Action Count: {self.tea_action_count} | "
                f"Activation Ratio: {activation_ratio:.2f}"
            )

            # ========== Auto Reset on Episode Done ==========
            if self.auto_reset_on_done:
                self.get_logger().info('Auto reset triggered on episode done')
                self.reset_episode()
            else:
                # Manual reset of internal states
                self.current_episode_reward = 0.0
                self.current_episode_length = 0
                self.obstacle_reward = 0.0
                self.goal_distance_reward = 0.0
                self.body_state_reward = 0.0
                self.stu_action_count = 0
                self.tea_action_count = 0                        
            
        
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
            os.makedirs(os.path.dirname(f"{MODEL_PATH}_{self.update_count}"), exist_ok=True)
            self.agent.save_models(f"{MODEL_PATH}_{self.update_count}")
            
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
