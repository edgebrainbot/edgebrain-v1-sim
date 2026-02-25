#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud
import numpy as np
import struct

import tf2_ros
from scipy.spatial.transform import Rotation as R

class LidarToLegodom(Node):
    def __init__(self):
        super().__init__("lidar_to_legodom")

        self.declare_parameter('target_frame', 'leg_odom')
        self.declare_parameter('input_topic', '/scan/points')
        self.declare_parameter('output_topic', '/registered_scan')
        self.declare_parameter('timeout', 0.1)  # TF 查找超时时间(秒)
        
        self.target_frame = self.get_parameter('target_frame').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.tf_timeout = self.get_parameter('timeout').value

        # TF Buffer 和 Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            PointCloud2, 
            input_topic, 
            self.callback, 
            10
        )

        self.pub = self.create_publisher(
            PointCloud2, 
            output_topic, 
            10
        )

        self.success_count = 0
        self.fail_count = 0
        self.last_log_time = self.get_clock().now()

        self.get_logger().info(
            f'点云转换节点已启动: {input_topic} → {output_topic} '
            f'(目标坐标系: {self.target_frame})'
        )

    def parse_gazebo_cloud(self, msg):
        """从 Gazebo 的 PointCloud2 解析 Nx3 xyz numpy 数组"""
        step = msg.point_step
        data = msg.data
        N = msg.width

        if N == 0:
            return np.array([]).reshape(0, 3)

        pts = np.zeros((N, 3), dtype=np.float32)

        # XYZ offsets (假设 XYZRGB 或 XYZ 格式)
        ox, oy, oz = 0, 4, 8

        for i in range(N):
            base = i * step
            pts[i, 0] = struct.unpack_from('f', data, base + ox)[0]
            pts[i, 1] = struct.unpack_from('f', data, base + oy)[0]
            pts[i, 2] = struct.unpack_from('f', data, base + oz)[0]

        return pts

    def transform_to_matrix(self, transform):
        """将 TransformStamped 转换为 4x4 齐次变换矩阵"""
        t = transform.transform.translation
        q = transform.transform.rotation
        
        # 使用 scipy 的 Rotation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        rot_matrix = r.as_matrix()
        
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rot_matrix
        T[:3, 3] = [t.x, t.y, t.z]
        
        return T

    def callback(self, msg):
        # 1. 检查点云是否为空
        if msg.width == 0 or len(msg.data) == 0:
            self.get_logger().warn("收到空点云，跳过", throttle_duration_sec=5.0)
            return

        # 2. 解析点云
        try:
            pts = self.parse_gazebo_cloud(msg)
            if pts.shape[0] == 0:
                return
        except Exception as e:
            self.fail_count += 1
            self.get_logger().error(f"解析点云失败: {e}")
            return

        try:
            # 使用点云消息的时间戳
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,          # 目标坐标系
                msg.header.frame_id,        # 源坐标系
                msg.header.stamp,           # 使用点云的时间戳
                timeout=Duration(seconds=self.tf_timeout)
            )
        except tf2_ros.LookupException as e:
            self.fail_count += 1
            self.get_logger().warn(
                f"TF 查找失败: {e}",
                throttle_duration_sec=2.0
            )
            return
        except tf2_ros.ExtrapolationException as e:
            self.fail_count += 1
            self.get_logger().warn(
                f"TF 外推失败: {e}",
                throttle_duration_sec=2.0
            )
            return
        except Exception as e:
            self.fail_count += 1
            self.get_logger().error(f"TF 获取异常: {e}")
            return

        # 4. 转换为变换矩阵
        try:
            T = self.transform_to_matrix(transform)
        except Exception as e:
            self.fail_count += 1
            self.get_logger().error(f"变换矩阵计算失败: {e}")
            return

        # 5. 应用变换
        try:
            # 转为齐次坐标 (N, 4)
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
            
            # 变换: (4, 4) @ (N, 4).T = (4, N) -> (N, 4) -> 取前3列
            pts_transformed = (T @ pts_h.T).T[:, :3]

            # pts_transformed[:, 0] = -pts_transformed[:, 0]  # X 轴取反以适应 leg_odom 坐标系
            
        except Exception as e:
            self.fail_count += 1
            self.get_logger().error(f"点云变换失败: {e}")
            return

        # 6. 构造新的 PointCloud2 消息
        try:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]

            # 创建新 header,注意 frame_id 要与目标坐标系一致
            header = msg.header
            header.frame_id = 'map'

            cloud_msg = create_cloud(header, fields, pts_transformed.tolist())
            
            # 发布
            self.pub.publish(cloud_msg)
            
            self.success_count += 1
            
            # 定期输出统计信息
            now = self.get_clock().now()
            if (now - self.last_log_time).nanoseconds > 5e9:  # 每5秒
                self.get_logger().info(
                    f'转换统计 - 成功: {self.success_count}, '
                    f'失败: {self.fail_count}, '
                    f'当前点数: {pts_transformed.shape[0]}'
                )
                self.last_log_time = now
                
        except Exception as e:
            self.fail_count += 1
            self.get_logger().error(f"构造点云消息失败: {e}")
            return

def main():
    rclpy.init()
    node = LidarToLegodom()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'节点退出 - 总成功: {node.success_count}, '
            f'总失败: {node.fail_count}'
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()