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
from edge_drl_student.memory.buffers.off_policy.cognitive_replay_buffer import CognitiveReplayBuffer
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.runners.distributed_runner import (
    DistributedRunner,
    TopicConfig,
    LoopConfig,
    TrainingConfig,
    ResidualConfig
)
from edge_drl_student.common.reset_manager import ResetManager
from edge_drl_student.common.episode_handler import EpisodeManager
from edge_drl_student.processor.vision_processor import VisionProcessor


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
        
        # ========== Dimensions ==========
        self.body_state_dim = 12
        self.vision_feature_dim = 24
        self.obs_dim = self.body_state_dim + self.vision_feature_dim
        self.act_dim = 6
        
        self.get_logger().info(
            f"📐 Dimensions: obs_dim={self.obs_dim}, act_dim={self.act_dim}"
        )
        
        # ========== Initialize Logger ==========
        self.logger = Logger(log_dir=LOG_PATH, device=self.device)
        self.get_logger().info(f"📊 Logger initialized: {LOG_PATH}")
        
        # ========== Initialize Replay Buffer ==========
        batch_size = 512
        if RUN == 'dual':
            self.buffer = DualReplayBuffer(
                student_buf_size=int(1e5),
                student_prefill_size=batch_size,
                teacher_buf_size=int(1e5),
                teacher_prefill_size=50,
                obs_dim=self.obs_dim,
                action_dim=self.act_dim,
                device=self.device
            )
        elif RUN == 'cognitive':
            self.buffer = CognitiveReplayBuffer(
                student_buf_size=int(1e5),
                student_prefill_size=batch_size,
                teacher_buf_size=int(1e5),
                teacher_prefill_size=50,
                obs_dim=self.obs_dim,
                action_dim=self.act_dim,
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported RUN type: {RUN}")
        
        self.get_logger().info(f"✅ Replay buffer initialized: {RUN}")
        
        # ========== Initialize Components ==========
        
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
        )
        
        # Reset Manager
        self.reset_manager = ResetManager(
            node=self,
            reset_timeout=15.0,
            logger=self.logger,
            reset_service_name='/edge/reset',
            coordinator_service_name='/edge/reset/coordinator'
        )
        
        # ========== Agent Configuration ==========
        
        # Topic configuration
        topic_cfg = TopicConfig(
            transition='transition',
            tracking_error='tracking_error_stu',
            action='drl_student_action',
            learned_action='drl_student_learned_action',
            fall_detection='/edge/is_fall'
        )
        
        # Loop configuration
        loop_cfg = LoopConfig(
            action_frequency=10.0,   # 10Hz
            learning_frequency=2.0   # 2Hz
        )
        
        # Training configuration
        train_cfg = TrainingConfig(
            batch_size=batch_size,
            save_interval=500,
            sync_interval=5,
            log_interval=100,
            model_save_path=MODEL_PATH
        )
        
        # Residual control configuration
        residual_gain = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 35.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=torch.float32, device=self.device) * -0.1
        
        residual_cfg = ResidualConfig(
            residual_gain=residual_gain,
            policy_blending_weight=0.3
        )
        
        # Action bounds
        action_bound = (
            torch.tensor([-10., -10., -10., -20., -20., -20.], device=self.device),
            torch.tensor([ 10.,  10.,  10.,  20.,  20.,  20.], device=self.device)
        )
        
        # ⭐ Initialize Distributed Runner
        self.agent = DistributedRunner(
            # Core parameters
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            node=self,
            logger=self.logger,
            replay_buffer=self.buffer,
            action_bound=action_bound,
            body_state_dim=self.body_state_dim,
            device=self.device,
            
            # Configuration objects
            topic_config=topic_cfg,
            loop_config=loop_cfg,
            training_config=train_cfg,
            residual_config=residual_cfg,
            
            # Components
            vision_processor=self.vision_processor,
            episode_manager=self.episode_manager,
            reset_manager=self.reset_manager,
            
            # Options
            auto_reset_on_fall=True,
            
            # Agent hyperparameters
            actor_lr=3e-4,
            critic_lr=3e-4,
            gamma=0.99,
            tau=0.005,
            actor_hidden_dims=(256, 256, 256),
            critic_hidden_dims=(256, 256, 256)
        )
        
        self.get_logger().info("✅ Distributed Runner initialized")
        
        # ========== ROS2 Subscriptions ==========
        # ⭐ Only subscribe to PointCloud2 here
        # Other subscriptions are handled by DistributedRunner
        self.create_subscription(
            PointCloud2,
            '/registered_scan',
            self.vision_processor.pointcloud_callback,
            10
        )
        
        # ========== ROS2 Services ==========
        self.create_service(Trigger, 'manual_reset', self.manual_reset_callback)
        self.get_logger().info(
            f"✅ Manual reset service created: /edge/manual_reset\n"
            f"   Usage: ros2 service call /edge/manual_reset std_srvs/srv/Trigger"
        )
        
        # ⭐ Start agent loops
        self.agent.start()
        
        # ========== Log Initialization Complete ==========
        # self._log_initialization_complete(episode_cfg, loop_cfg, train_cfg)
    
    # ==================== Callbacks ====================
    
    def manual_reset_callback(self, request: Trigger.Request, response: Trigger.Response):
        """
        Manual reset service callback
        
        Usage:
            ros2 service call /edge/manual_reset std_srvs/srv/Trigger
        """
        self.get_logger().info('🔄 Manual reset triggered by service call')
        
        if self.reset_manager.is_resetting:
            response.success = False
            response.message = 'Reset already in progress, please wait'
            return response
        
        # Perform manual reset
        success, message = self.reset_manager.call_reset_service(ResetReason.MANUAL)
        
        response.success = success
        response.message = (
            f'Manual reset {"successful" if success else "failed"}. '
            f'Episode #{self.episode_manager.episode_count}. '
            f'{message}'
        )
        
        return response
    
    # ==================== Logging ====================
    
    def _log_initialization_complete(self, episode_cfg, loop_cfg, train_cfg):
        """Log initialization complete message"""
        self.get_logger().info(
            f"\n{'='*70}\n"
            f"🚀 DRL Student Node Started\n"
            f"{'='*70}\n"
            f"📦 Project:       {PROJECT_NAME}\n"
            f"🎮 Run Mode:      {RUN}\n"
            f"💻 Device:        {self.device}\n"
            f"💾 Model Path:    {MODEL_PATH}\n"
            f"📊 Log Path:      {LOG_PATH}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"📐 Dimensions:\n"
            f"   Obs Dim:     {self.obs_dim} (body={self.body_state_dim} + vision={self.vision_feature_dim})\n"
            f"   Act Dim:     {self.act_dim}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"⚙️  Configuration:\n"
            f"   Action Freq:        {loop_cfg.action_frequency}Hz\n"
            f"   Learning Freq:      {loop_cfg.learning_frequency}Hz\n"
            f"   Batch Size:         {train_cfg.batch_size}\n"
            f"   Save Interval:      {train_cfg.save_interval}\n"
            f"   Sync Interval:      {train_cfg.sync_interval}\n"
            f"   Max Episode Length: {episode_cfg.max_episode_length}\n"
            f"   Success Threshold:  {episode_cfg.success_reward_threshold}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
            f"📡 ROS2 Topics:\n"
            f"   Subscriptions:\n"
            f"     - /registered_scan (PointCloud2)\n"
            f"     - /transition (Transition)\n"
            f"     - /tracking_error_stu (Vec12Msg)\n"
            f"     - /edge/is_fall (Bool)\n"
            f"   Publishers:\n"
            f"     - /drl_student_action (EdgeAction)\n"
            f"     - /drl_student_learned_action (EdgeAction)\n"
            f"   Services:\n"
            f"     - /edge/manual_reset (Trigger)\n"
            f"{'='*70}\n"
        )
    
    # ==================== Cleanup ====================
    
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
