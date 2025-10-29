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

from edge_msgs.msg import Experience, TrackingError, Action
from edge_drl_student.memory.buffers.replay_buffer.solo_replay_buffer import SoloReplayBuffer

class SimplePolicy(nn.Module):
    def __init__(self, obs_dim=12, act_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


MODEL_PATH = "/home/edgebot/models/policy.pt"


class DRLStudentNode(Node):
    def __init__(self):
        super().__init__('drl_student', namespace='edge')

        # Initialize policy network and optimizer
        self.policy = SimplePolicy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        self.obs_dim, self.act_dim = 12, 6
        self.buffer = SoloReplayBuffer(buffer_size=int(1e5), obs_dim=self.obs_dim, action_dim=self.act_dim, device='cpu')
        self.batch_size = 32
        self.last_reload_time = 0.0

        # ros2 topics
        self.action_pub = self.create_publisher(Action, 'drl_student_action', 10)
        # self.create_subscription(TrackingError, '/tracking_error', self.tracking_error_callback, 10)
        self.create_subscription(Experience, 'transition', self.transition_callback, 10)

        # Create timers
        self.action_timer = self.create_timer(0.1, self.action_loop)  # Generate and publish actions
        self.learn_timer = self.create_timer(1.0, self.learn_step)    # Learning weights updates

        self.get_logger().info("✅ DRLStudentNode initialized")

    def action_loop(self):
        
        msg = Action()
        msg.action = np.array([0.0]*self.act_dim, dtype=np.float64)
        self.action_pub.publish(msg)

        # self.get_logger().info(f"[Action] Published action (r={msg.action:.3f})")

    def transition_callback(self, msg: Experience):
        t = self.buffer.to_transition(
            obs=msg.obs,
            action=msg.action,
            reward=msg.reward,
            next_obs=msg.next_obs,
            done=msg.done
        )
        self.buffer.add_transition(t)

    def learn_step(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        obs_batch = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        act_batch = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
        target = -torch.norm(act_batch, dim=1, keepdim=True)

        pred = self.policy(obs_batch)
        loss = self.loss_fn(pred.mean(dim=1, keepdim=True).detach().clone(), target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
