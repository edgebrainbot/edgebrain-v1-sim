#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Empty, Trigger
from geometry_msgs.msg import Pose
import subprocess
import time


class ResetHandlerNode(Node):
    def __init__(self):
        super().__init__('reset_handler_node')
        
        # Use reentrant callback group to allow processing other callbacks during service calls
        self.callback_group = ReentrantCallbackGroup()
        
        # ========== Parameter Declaration ==========
        self.declare_parameter('world_name', 'bayland_world')
        self.declare_parameter('robot_name', 'go2')
        self.declare_parameter('reset_position_x', 0.0)
        self.declare_parameter('reset_position_y', 0.0)
        self.declare_parameter('reset_position_z', 0.35)
        self.declare_parameter('reset_orientation_x', 0.0)
        self.declare_parameter('reset_orientation_y', 0.0)
        self.declare_parameter('reset_orientation_z', 0.0)
        self.declare_parameter('reset_orientation_w', 1.0)
        self.declare_parameter('service_timeout', 3000)
        self.declare_parameter('reset_world', False)  # Whether to reset entire world
        self.declare_parameter('reset_robot_pose', True)  # Whether to reset robot pose
        self.declare_parameter('wait_after_reset', 0.5)  # Wait time after reset (seconds)
        
        # ========== Get Parameters ==========
        self.world_name = self.get_parameter('world_name').value
        self.robot_name = self.get_parameter('robot_name').value
        self.reset_pos_x = self.get_parameter('reset_position_x').value
        self.reset_pos_y = self.get_parameter('reset_position_y').value
        self.reset_pos_z = self.get_parameter('reset_position_z').value
        self.reset_ori_x = self.get_parameter('reset_orientation_x').value
        self.reset_ori_y = self.get_parameter('reset_orientation_y').value
        self.reset_ori_z = self.get_parameter('reset_orientation_z').value
        self.reset_ori_w = self.get_parameter('reset_orientation_w').value
        self.timeout = self.get_parameter('service_timeout').value
        self.reset_world_flag = self.get_parameter('reset_world').value
        self.reset_robot_pose_flag = self.get_parameter('reset_robot_pose').value
        self.wait_after_reset = self.get_parameter('wait_after_reset').value
        
        # ========== Create Services ==========
        # Service 1: /reset_episode (Empty)
        self.reset_srv = self.create_service(
            Empty,
            'reset_episode',
            self.reset_callback,
            callback_group=self.callback_group
        )
        
        # Service 2: /reset_episode_trigger (Trigger with feedback)
        self.reset_trigger_srv = self.create_service(
            Trigger,
            'reset_episode_trigger',
            self.reset_trigger_callback,
            callback_group=self.callback_group
        )
        
        # ========== NEW: Service 3: /edge/reset (Trigger) ==========
        self.edge_reset_srv = self.create_service(
            Trigger,
            '/edge/reset',
            self.edge_reset_callback,
            callback_group=self.callback_group
        )
        
        # ========== Gazebo Service Clients ==========
        self.gz_reset_client = self.create_client(
            Empty,
            '/reset_world',
            callback_group=self.callback_group
        )
        
        # ========== Statistics ==========
        self.reset_count = 0
        self.last_reset_time = None
        
        # ========== Log Output ==========
        self.get_logger().info('=' * 60)
        self.get_logger().info('Reset Handler Node Initialized')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'World: {self.world_name}')
        self.get_logger().info(f'Robot: {self.robot_name}')
        self.get_logger().info(f'Reset Position: [{self.reset_pos_x:.2f}, {self.reset_pos_y:.2f}, {self.reset_pos_z:.2f}]')
        self.get_logger().info(f'Reset Orientation: [{self.reset_ori_x:.2f}, {self.reset_ori_y:.2f}, {self.reset_ori_z:.2f}, {self.reset_ori_w:.2f}]')
        self.get_logger().info(f'Reset World: {self.reset_world_flag}')
        self.get_logger().info(f'Reset Robot Pose: {self.reset_robot_pose_flag}')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Available Services:')
        self.get_logger().info('  - /reset_episode (std_srvs/srv/Empty)')
        self.get_logger().info('  - /reset_episode_trigger (std_srvs/srv/Trigger)')
        self.get_logger().info('  - /edge/reset (std_srvs/srv/Trigger)')  # ← NEW
        self.get_logger().info('=' * 60)
    
    def reset_callback(self, request, response):
        """Empty service callback (compatibility interface)"""
        success, message = self._perform_reset()
        if not success:
            self.get_logger().error(f'Reset failed: {message}')
        return response
    
    def reset_trigger_callback(self, request, response):
        """Trigger service callback (recommended, with detailed feedback)"""
        success, message = self._perform_reset()
        response.success = success
        response.message = message
        return response
    
    def edge_reset_callback(self, request, response):
        """
        /edge/reset service callback
        This is called by DRL student node to reset Gazebo
        """
        self.get_logger().info('🔄 [/edge/reset] Reset request received from DRL student')
        success, message = self._perform_reset()
        response.success = success
        response.message = message
        return response
    
    def _perform_reset(self):
        """
        Core logic for performing reset operation
        
        Returns:
            tuple: (success: bool, message: str)
        """
        start_time = time.time()
        self.reset_count += 1
        
        self.get_logger().info(f'[Reset #{self.reset_count}] Starting reset sequence...')
        
        try:
            # ========== Step 1: Reset Gazebo World ==========
            if self.reset_world_flag:
                world_reset_success = self._reset_gazebo_world()
                if not world_reset_success:
                    return False, 'Failed to reset Gazebo world'
                
                # Wait for world to stabilize
                time.sleep(self.wait_after_reset)
            
            # ========== Step 2: Reset Robot Pose ==========
            if self.reset_robot_pose_flag:
                pose_reset_success = self._reset_robot_pose()
                if not pose_reset_success:
                    return False, 'Failed to reset robot pose'
                
                # Wait for robot to stabilize
                time.sleep(self.wait_after_reset)
            
            # ========== Reset Successful ==========
            elapsed_time = time.time() - start_time
            self.last_reset_time = time.time()
            
            success_msg = (
                f'Reset #{self.reset_count} completed successfully in {elapsed_time:.2f}s'
            )
            self.get_logger().info(success_msg)
            
            return True, success_msg
            
        except Exception as e:
            error_msg = f'Exception during reset: {str(e)}'
            self.get_logger().error(error_msg)
            return False, error_msg
    
    def _reset_gazebo_world(self):
        """
        Reset Gazebo world
        
        Returns:
            bool: Whether successful
        """
        self.get_logger().info('  [1/2] Resetting Gazebo world...')
        
        # Method 1: Try using ROS2 service (if ros_gz_bridge is available)
        if self.gz_reset_client.wait_for_service(timeout_sec=1.0):
            try:
                self.get_logger().debug('    Using ROS2 /reset_world service')
                future = self.gz_reset_client.call_async(Empty.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                
                if future.result() is not None:
                    self.get_logger().info('    ✓ World reset via ROS2 service')
                    return True
                else:
                    self.get_logger().warn('    ROS2 service call failed, trying ign command...')
            except Exception as e:
                self.get_logger().warn(f'    ROS2 service error: {e}, trying ign command...')
        
        # Method 2: Use Ignition Gazebo command line
        try:
            self.get_logger().debug('    Using ign service command')
            
            service_name = f'/world/{self.world_name}/control'
            cmd = [
                'ign', 'service',
                '-s', service_name,
                '--reqtype', 'ignition.msgs.WorldControl',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', str(self.timeout),
                '--req', 'reset: {all: true}'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout / 1000.0 + 2.0
            )
            
            if result.returncode == 0:
                self.get_logger().info('    ✓ World reset via ign command')
                return True
            else:
                self.get_logger().error(f'    ✗ ign command failed: {result.stderr}')
                return False
                
        except subprocess.TimeoutExpired:
            self.get_logger().error('    ✗ ign command timeout')
            return False
        except Exception as e:
            self.get_logger().error(f'    ✗ ign command exception: {e}')
            return False
    
    def _reset_robot_pose(self):
        """
        Reset robot pose to initial position
        
        Returns:
            bool: Whether successful
        """
        self.get_logger().info('  [2/2] Resetting robot pose...')
        
        try:
            service_name = f'/world/{self.world_name}/set_pose'
            
            pose_msg = (
                f"name: '{self.robot_name}', "
                f"position: {{x: {self.reset_pos_x}, y: {self.reset_pos_y}, z: {self.reset_pos_z}}}, "
                f"orientation: {{x: {self.reset_ori_x}, y: {self.reset_ori_y}, "
                f"z: {self.reset_ori_z}, w: {self.reset_ori_w}}}"
            )
            
            cmd = [
                'ign', 'service',
                '-s', service_name,
                '--reqtype', 'ignition.msgs.Pose',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', str(self.timeout),
                '--req', pose_msg
            ]
            
            self.get_logger().debug(f'    Calling: {service_name}')
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout / 1000.0 + 2.0
            )
            
            if result.returncode == 0:
                self.get_logger().info(
                    f'    ✓ Robot pose reset to '
                    f'[{self.reset_pos_x:.2f}, {self.reset_pos_y:.2f}, {self.reset_pos_z:.2f}]'
                )
                return True
            else:
                self.get_logger().error(f'    ✗ Pose reset failed: {result.stderr}')
                return False
                
        except subprocess.TimeoutExpired:
            self.get_logger().error('    ✗ Pose reset timeout')
            return False
        except Exception as e:
            self.get_logger().error(f'    ✗ Pose reset exception: {e}')
            return False
    
    def get_reset_stats(self):
        """Get reset statistics"""
        return {
            'reset_count': self.reset_count,
            'last_reset_time': self.last_reset_time
        }


def main(args=None):
    rclpy.init(args=args)
    
    node = ResetHandlerNode()
    
    # Use multi-threaded executor to support concurrent service calls
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Reset Handler Node')
        stats = node.get_reset_stats()
        node.get_logger().info(f'Total resets performed: {stats["reset_count"]}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
