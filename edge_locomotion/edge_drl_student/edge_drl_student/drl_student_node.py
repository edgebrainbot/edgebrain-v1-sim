#!/usr/bin/env python3
"""
DRL Student Node - Clean and Simplified
"""

import os
import rclpy
import torch
from rclpy.node import Node
from datetime import datetime
from std_srvs.srv import Trigger
from sensor_msgs.msg import PointCloud2

from edge_drl_student.common.logger import Logger
from edge_drl_student.utils.utils import ResetReason
from edge_drl_student.common.reset_manager import ResetManager
from edge_drl_student.common.episode_handler import EpisodeManager
from edge_drl_student.processor.vision_processor import VisionProcessor
from edge_drl_student.runners.distributed_runner import DistributedRunner
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.memory.buffers.off_policy.cognitive_replay_buffer import CognitiveReplayBuffer

# ========== Configuration ==========
RUN = 'cognitive'  # 'cognitive' or 'dual'

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PROJECT_NAME = f"drl_student_{RUN}_{TIMESTAMP}"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "models", PROJECT_NAME)
LOG_PATH = os.path.join(CURRENT_DIR, "logs", PROJECT_NAME)

torch.set_printoptions(sci_mode=False)


class DRLStudentNode(Node):
    
    def __init__(self):
        super().__init__(PROJECT_NAME, namespace='edge')
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.declare_parameters(
            namespace="",
            parameters=[
                ("MATRIX_P", rclpy.Parameter.Type.DOUBLE_ARRAY),
                ("IDENTITY_MAT", rclpy.Parameter.Type.DOUBLE_ARRAY),
                ("SAFETY_ENVELOPE_P", rclpy.Parameter.Type.DOUBLE_ARRAY),
                ("RESIDUAL_GAIN", rclpy.Parameter.Type.DOUBLE_ARRAY),
                ("DEFAULT_KP", rclpy.Parameter.Type.DOUBLE_ARRAY),
                ("DEFAULT_KD", rclpy.Parameter.Type.DOUBLE_ARRAY),
            ]
        )
        
        RESIDUAL_GAIN = self.get_parameter("RESIDUAL_GAIN").value
        
        # ========== Dimensions ==========
        self.act_dim = 6
        self.body_state_dim = 12
        self.vision_feature_dim = 24
        self.obs_dim = self.body_state_dim + self.vision_feature_dim
        self.get_logger().info(f"📐 Dimensions: obs_dim={self.obs_dim}, act_dim={self.act_dim}")
        
        # ========== Initialize Logger ==========
        self.logger = Logger(log_dir=LOG_PATH, device=self.device)
        self.get_logger().info(f"📊 Logger initialized: {LOG_PATH}")

        # ========== Initialize Replay Buffer ==========
        self.batch_size = 100
        replay_buffer_cfg = {
            'student_buf_size': int(1e5),
            'student_prefill_size': self.batch_size,
            'teacher_buf_size': int(1e5),
            'teacher_prefill_size': 50,
            'obs_dim': self.obs_dim,
            'action_dim': self.act_dim,
            'device': self.device
        }
        if RUN == 'dual':
            self.buffer = DualReplayBuffer(**replay_buffer_cfg)
        elif RUN == 'cognitive':
            self.buffer = CognitiveReplayBuffer(**replay_buffer_cfg)
        else:
            raise ValueError(f"Unsupported RUN type: {RUN}")
        self.get_logger().info(f"✅ Replay buffer initialized: {RUN}")
        
        # Vision Processor
        self.vision_processor = VisionProcessor(
            node=self,
            topic_name='/registered_scan',
            queue_size=10,
            device=self.device,
            feature_dim=self.vision_feature_dim
        )

        # Episode Manager        
        self.episode_manager = EpisodeManager(
            logger=self.logger,
            max_episode_length=int(5e4),
            training_episode=10000,
            moving_average_window=100
        )
        
        # Reset Manager
        self.reset_manager = ResetManager(
            node=self,
            reset_timeout=15.0,
            logger=self.logger,
            reset_service_name='/edge/reset',
            coordinator_service_name='/edge/reset/coordinator'
        )
        
        # Action bounds
        # action_bound = (
        #     torch.tensor([-10., -10., -10., -20., -20., -20.], device=self.device),
        #     torch.tensor([ 10.,  10.,  10.,  20.,  20.,  20.], device=self.device)
        # )
        action_bound = (-10, 10)  # Uniform bound for all actions
        
        ros2_topic_name = {
            'transition_topic': '/edge/transition',
            'drl_action_topic': '/edge/drl_student_action',
            'tracking_error_topic': '/edge/tracking_error_stu',
            'drl_learned_action_topic': '/edge/drl_student_learned_action',
            'fall_detection_topic': '/edge/is_fall',
        }
        
        ros2_service_name = {
            'manual_reset_service': '/edge/manual_reset',
        }
        
        training_cfg = {
            'batch_size': self.batch_size,
            'log_interval': 100,
            'sync_interval': 5,
            'save_interval': 500,
            'normalize_obs': True,
            'normalize_reward': True,
            'auto_reset_on_fall': True,
        }
        
        agent_cfg = {
            'model_save_path': MODEL_PATH,
        }
        # ⭐ Initialize Distributed Runner
        self.agent = DistributedRunner(
            node=self,
            logger=self.logger,
            device=self.device,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            body_state_dim=self.body_state_dim,
            
            reset_manager=self.reset_manager,
            episode_manager=self.episode_manager,
            vision_processor=self.vision_processor,
            replay_buffer=self.buffer,
            action_bound=action_bound,
            
            residual_gain=torch.tensor(RESIDUAL_GAIN, device=self.device).reshape(
                self.act_dim, self.body_state_dim),
            
            **ros2_topic_name,
            **ros2_service_name,
            
            **training_cfg,
            model_save_path=MODEL_PATH,
            # agent_kwargs=agent_cfg
        )
        
        self.get_logger().info("✅ Distributed Runner initialized")
        
        # ⭐ Start agent loops
        self.agent.start()
        
        self.get_logger().info("🚀 DRL Student Node is up and running!")
    
    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info("🛑 Shutting down DRL Student Node...")
        
        # Stop agent loops
        self.agent.stop()
        
        # Save final checkpoint
        self.agent.save_final_checkpoint()
        
        # Print final statistics
        self._print_final_statistics()
        
        super().destroy_node()
    
    def _print_final_statistics(self):
        """Print final statistics"""
        stats = self.agent.get_agent_stats()
        perf_stats = self.agent.get_performance_stats()
        sync_stats = self.agent.get_sync_stats()
        reset_stats = self.agent.get_reset_stats()
        fall_stats = self.agent.get_fall_stats()
        global_stats = self.episode_manager.get_global_statistics()
        
        self.get_logger().info(
            f"\n{'='*70}\n"
            f"📊 Final Statistics\n"
            f"{'='*70}\n"
            f"🎓 Training:\n"
            f"   Total Updates:       {stats['update_count']}\n"
            f"   Total Syncs:         {self.agent.actor_learner.sync_count}\n"
            f"   Total Transitions:   {stats['transition_count']}\n"
            f"   Total Actions:       {stats['action_count']}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"📉 Losses:\n"
            f"   Avg Actor Loss:      {stats['avg_actor_loss']:.4f}\n"
            f"   Avg Critic Loss:     {stats['avg_critic_loss']:.4f}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"⚡ Performance:\n"
            f"   Avg Action Time:     {perf_stats['action_time_ms']:.2f}ms\n"
            f"   Avg Learn Time:      {perf_stats['learn_time_ms']:.2f}ms\n"
            f"   Avg Sync Time:       {sync_stats['avg_sync_time']*1000:.2f}ms\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"🎬 Episodes:\n"
            f"   Total Episodes:      {global_stats['episode_count']}\n"
            f"   Total Steps:         {global_stats['total_steps']}\n"
            f"   Success Rate:        {global_stats['success_rate']:.2%}\n"
            f"   Avg Reward:          {global_stats['average_reward']:.2f}\n"
            f"   Avg Length:          {global_stats['average_length']:.1f}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"🔄 Resets:\n"
            f"   Total Resets:        {reset_stats['total']}\n"
            f"   Success Rate:        {reset_stats['success_rate']:.2%}\n"
            f"   Episode Done:        {reset_stats['reasons']['episode_done']}\n"
            f"   Fall Detected:       {reset_stats['reasons']['fall_detected']}\n"
            f"   Timeout:             {reset_stats['reasons']['timeout']}\n"
            f"   Manual:              {reset_stats['reasons']['manual']}\n"
            f"   External:            {reset_stats['reasons']['external']}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"⚠️  Falls:\n"
            f"   Total Falls:         {fall_stats['fall_count']}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"💾 Buffer Status:\n"
            f"   Student Size:        {len(self.buffer.stu_buf)}/{self.buffer.stu_buf.capacity}\n"
            f"   Teacher Size:        {len(self.buffer.tea_buf)}/{self.buffer.tea_buf.capacity}\n"
            f"{'='*70}\n"
        )


def main(args=None):
    """Main entry point"""
    print(f"\n{'='*70}")
    print(f"🚀 Starting DRL Student Node")
    print(f"{'='*70}")
    print(f"  Project:    {PROJECT_NAME}")
    print(f"  Model Path: {MODEL_PATH}")
    print(f"  Log Path:   {LOG_PATH}")
    print(f"{'='*70}\n")
    
    rclpy.init(args=args)
    node = DRLStudentNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("⌨️  Keyboard interrupt received")
    except Exception as e:
        node.get_logger().error(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("\n✅ DRL Student Node shutdown complete\n")


if __name__ == '__main__':
    main()
