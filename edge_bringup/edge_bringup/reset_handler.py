#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import subprocess

class ResetHandlerNode(Node):
    def __init__(self):
        super().__init__('reset_handler_node')
        
        # 创建 reset service
        self.reset_srv = self.create_service(
            Empty, 
            'reset_episode', 
            self.reset_callback
        )
        
        # Gazebo reset service client
        self.gz_reset_client = self.create_client(Empty, '/reset_world')
        
        # 机器人初始位置
        self.initial_pose = Pose()
        self.initial_pose.position.x = 0.0
        self.initial_pose.position.y = 0.0
        self.initial_pose.position.z = 0.5
        self.initial_pose.orientation.w = 1.0
        
        self.get_logger().info('Reset Handler Node initialized')
    
    def reset_callback(self, request, response):
        self.get_logger().info('Resetting episode...')
        
        # 方法1: 使用 Gazebo service (如果有 ros_gz_bridge)
        if self.gz_reset_client.wait_for_service(timeout_sec=1.0):
            future = self.gz_reset_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        else:
            # 方法2: 直接调用 ign 命令
            try:
                subprocess.run([
                    'ign', 'service', '-s', '/world/bayland_world/control',
                    '--reqtype', 'ignition.msgs.WorldControl',
                    '--reptype', 'ignition.msgs.Boolean',
                    '--timeout', '3000',
                    '--req', 'reset: {all: true}'
                ], check=True, timeout=5)
                self.get_logger().info('Gazebo world reset successfully')
            except Exception as e:
                self.get_logger().error(f'Failed to reset Gazebo: {e}')
        
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ResetHandlerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
