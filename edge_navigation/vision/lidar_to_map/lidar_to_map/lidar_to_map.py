#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud
import numpy as np
import struct

class LidarXYFlip(Node):
    def __init__(self):
        super().__init__("lidar_xy_flip")

        self.declare_parameter('input_topic', '/scan/points')
        self.declare_parameter('output_topic', '/registered_scan')

        input_topic  = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.sub = self.create_subscription(PointCloud2, input_topic,  self.callback, 10)
        self.pub = self.create_publisher(PointCloud2,    output_topic, 10)

        self.get_logger().info(f'lidar_xy_flip: {input_topic} → {output_topic}  (-X, -Y)')

    def callback(self, msg: PointCloud2):
        if msg.width == 0 or len(msg.data) == 0:
            return

        N    = msg.width
        step = msg.point_step
        data = msg.data

        pts = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            base = i * step
            pts[i, 0] = struct.unpack_from('f', data, base + 0)[0]
            pts[i, 1] = struct.unpack_from('f', data, base + 4)[0]
            pts[i, 2] = struct.unpack_from('f', data, base + 8)[0]

        # 翻转 X 和 Y
        pts[:, 0] = -pts[:, 0]
        pts[:, 1] = -pts[:, 1]

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        out = create_cloud(msg.header, fields, pts.tolist())
        self.pub.publish(out)

def main():
    rclpy.init()
    node = LidarXYFlip()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()