#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import random

from edge_drl_student.utils.utils import ActionMode
from edge_msgs.msg import Transition, TrackingError, EdgeAction
from edge_drl_student.memory.buffers.off_policy.solo_replay_buffer import SoloReplayBuffer
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.memory.buffers.off_policy.cognitive_replay_buffer import CognitiveReplayBuffer
from edge_drl_student.agents.off_policy.ddpg_agent import DDPGAgent
from ament_index_python.packages import get_package_prefix

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "models", "drl_student")


class DRLStudentNode(Node):
    def __init__(self):
        super().__init__('drl_student', namespace='edge')

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.obs_dim = 12
        self.act_dim = 6
        self.policy_blendering_weight = 0.5
        # self.buffer = DualReplayBuffer(student_buf_size=int(1e5), 
        #                                student_prefill_size=int(1e3),
        #                                teacher_buf_size=int(1e5), 
        #                                teacher_prefill_size=int(1),
        #                                obs_dim=self.obs_dim, 
        #                                action_dim=self.act_dim, 
        #                                device=self.device)
        self.buffer = CognitiveReplayBuffer(student_buf_size=int(1e4), 
                                            student_prefill_size=int(1e3),
                                            teacher_buf_size=int(1e4), 
                                            teacher_prefill_size=int(1),
                                            obs_dim=self.obs_dim, 
                                            action_dim=self.act_dim, 
                                            device=self.device)
        self.batch_size = 512
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
        
        edge_action_msg = EdgeAction()
        
        edge_action_msg.header.stamp = self.get_clock().now().to_msg()
        edge_action_msg.control_action = self.output_action().detach().cpu().numpy().astype(np.float64) # Scaled control action
        edge_action_msg.action_mode = ActionMode.STUDENT.value
        
        self.action_pub.publish(edge_action_msg)

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
        # self.get_logger().info(f"[TrackingError] Received tracking error: {msg}")

    def learn_step(self):

        if not self.buffer.stu_buf.is_ready:
            self.get_logger().info(f"Student Buffer not ready for learning. Current size: {self.buffer.stu_buf_size}, required prefill size: {self.buffer.stu_buf.prefill_size}.")
            return
        
        if not self.buffer.tea_buf.is_ready:
            self.get_logger().info(f"Teacher Buffer not ready for learning. Current size: {self.buffer.tea_buf_size}, required prefill size: {self.buffer.tea_buf.prefill_size}.")
            return

        actor_loss, critic_loss = self.agent.update(self.batch_size)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        self.agent.save_models(MODEL_PATH)
        self.get_logger().info(f"[Learner] Updated policy (actor_loss={actor_loss.item():.4f}, critic_loss={critic_loss.item():.4f})")


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
