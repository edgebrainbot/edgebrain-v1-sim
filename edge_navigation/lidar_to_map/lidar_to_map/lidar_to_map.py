#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud
import numpy as np
import struct

import tf2_ros
import tf2_py
from geometry_msgs.msg import TransformStamped

class LidarToMap(Node):
    def __init__(self):
        super().__init__("lidar_to_legodom")

        # target = leg_odom wsseewsss
        self.target_frame = "leg_odom"

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            PointCloud2, "/scan/points", self.callback, 10
        )

        self.pub = self.create_publisher(
            PointCloud2, "/registered_scan", 10
        )

        self.get_logger().info("LidarLink → leg_odom 点云转换节点已启动")

    def parse_gazebo_cloud(self, msg):
        """从 Gazebo 的 PointCloudPacked 正确解析出 Nx3 xyz numpy 数组"""

        step = msg.point_step
        data = msg.data
        N = msg.width

        pts = np.zeros((N, 3), dtype=np.float32)

        # offsets
        ox = 0
        oy = 4
        oz = 8

        for i in range(N):
            base = i * step
            pts[i, 0] = struct.unpack_from('f', data, base + ox)[0]
            pts[i, 1] = struct.unpack_from('f', data, base + oy)[0]
            pts[i, 2] = struct.unpack_from('f', data, base + oz)[0]

        return pts

    def callback(self, msg):
        # 1. 解析点云
        if msg.width == 0:
            self.get_logger().warn("收到空点云，跳过")
            return

        try:
            pts = self.parse_gazebo_cloud(msg)
        except Exception as e:
            self.get_logger().error(f"解析点云失败: {e}")
            return

        # 2. Lookup TF: target_frame ← lidar frame
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF 获取失败: {str(e)}")
            return

        # tf 转矩阵
        t = tf.transform.translation
        q = tf.transform.rotation
        import tf_transformations
        T = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z

        # 3. 点云齐次变换
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_map = (T @ pts_h.T).T[:, :3]

        # 4. 构造新的 PointCloud2
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        header = msg.header
        header.frame_id = 'map'  # 改成 map 或 leg_odom

        cloud_msg = create_cloud(header, fields, pts_map)

        self.pub.publish(cloud_msg)

def main():
    rclpy.init()
    node = LidarToMap()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
