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
from edge_drl_student.agents.off_policy.ddpg_agent import DDPGAgent

class SimplePolicy(nn.Module):
    def __init__(self, obs_dim=12, act_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


MODEL_PATH = "/home/edgebot/models/policy.pt"


class DRLStudentNode(Node):
    def __init__(self):
        super().__init__('drl_student', namespace='edge')

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.obs_dim = 12
        self.act_dim = 6
        self.buffer = DualReplayBuffer(student_buf_size=int(1e5), 
                                       student_prefill_size=int(1e4),
                                       teacher_buf_size=int(1e5), 
                                       teacher_prefill_size=int(1e4),
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

        self.get_logger().info("✅ DRLStudentNode initialized")

    def action_loop(self):
        
        edge_action_msg = EdgeAction()
        
        edge_action_msg.header.stamp = self.get_clock().now().to_msg()
        edge_action_msg.control_action = self.agent.act(torch.tensor(
            np.array(self.tracking_error_.tracking_error, dtype=np.float32),
            dtype=torch.float32
        )).detach().numpy().astype(np.float64) # Scaled control action
        edge_action_msg.action_mode = ActionMode.STUDENT.value
        
        self.action_pub.publish(edge_action_msg)

        # self.get_logger().info(f"[Action] Published action (r={msg.action:.3f})")

    def transition_callback(self, msg: Transition):
        t = self.buffer.to_transition(
            obs=msg.obs,
            action=msg.action,
            reward=msg.reward,
            next_obs=msg.next_obs,
            done=msg.done
        )
        self.buffer.add_transition(t, action_mode=ActionMode.TEACHER)
        
    def tracking_error_callback(self, msg: TrackingError):
        self.tracking_error_ = msg
        self.get_logger().info(f"[TrackingError] Received tracking error: {msg}")

    def learn_step(self):

        if not self.buffer.stu_buf.is_ready:
            self.get_logger().info(f"[Learner] Buffer not ready for learning. Current size: {self.buffer.stu_buf_size}, required prefill size: {self.buffer.stu_buf.prefill_size}.")
            return
        
        if not self.buffer.tea_buf.is_ready:
            self.get_logger().info(f"[Learner] Buffer not ready for learning. Current size: {self.buffer.tea_buf_size}, required prefill size: {self.buffer.tea_buf.prefill_size}.")
            return

        self.agent.update(self.batch_size)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(self.policy.state_dict(), MODEL_PATH)
        self.get_logger().info(f"[Learner] Updated policy (loss={loss.item():.4f})")

    def maybe_reload_model(self):
        if time.time() - self.last_reload_time > 5.0 and os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location='cpu')
            self.policy.load_state_dict(state_dict)
            self.get_logger().info("Reloaded policy parameters 🔄")
            self.last_reload_time = time.time()


# ==========================================================
# 🔹 Main
# ==========================================================
def main(args=None):
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
