"""Transform simulated lidar returns into the map frame for terrain analysis."""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener


def transform_points(points, translation, quaternion):
    """Apply a rigid transform to an Nx3 array of finite points."""
    q = np.asarray(quaternion[:3], dtype=np.float64)
    cross = 2.0 * np.cross(q, points)
    return points + quaternion[3] * cross + np.cross(q, cross) + translation


class RegisteredScan(Node):
    def __init__(self):
        super().__init__('registered_scan_transform')
        input_topic = self.declare_parameter('input_topic', '/scan/points').value
        output_topic = self.declare_parameter('output_topic', '/registered_scan').value
        self.target_frame = self.declare_parameter('target_frame', 'map').value
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.publisher = self.create_publisher(PointCloud2, output_topic, 5)
        self.subscription = self.create_subscription(
            PointCloud2, input_topic, self.callback, qos_profile_sensor_data)

    def callback(self, msg):
        if not msg.width or not msg.height:
            return
        try:
            transform = self.buffer.lookup_transform(
                self.target_frame, msg.header.frame_id,
                rclpy.time.Time.from_msg(msg.header.stamp))
        except TransformException as exc:
            self.get_logger().warning(f'Waiting for lidar transform: {exc}', throttle_duration_sec=5)
            return
        fields = {field.name: field for field in msg.fields}
        if not all(name in fields and fields[name].datatype == PointField.FLOAT32
                   for name in ('x', 'y', 'z')):
            self.get_logger().error('Lidar cloud must contain FLOAT32 x, y and z fields')
            return
        endian = '>' if msg.is_bigendian else '<'
        def values(name):
            return np.ndarray((msg.height, msg.width), dtype=endian + 'f4',
                              buffer=msg.data, offset=fields[name].offset,
                              strides=(msg.row_step, msg.point_step)).reshape(-1)
        points = np.column_stack([values(name) for name in ('x', 'y', 'z')])
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        t = transform.transform.translation
        q = transform.transform.rotation
        points = transform_points(points, np.array([t.x, t.y, t.z]),
                                  [q.x, q.y, q.z, q.w])
        intensity = (values('intensity')[valid]
                     if 'intensity' in fields and fields['intensity'].datatype == PointField.FLOAT32
                     else np.zeros(len(points)))
        output_fields = [PointField(name=name, offset=i*4,
                                   datatype=PointField.FLOAT32, count=1)
                         for i, name in enumerate(('x', 'y', 'z', 'intensity'))]
        output = create_cloud(Header(stamp=msg.header.stamp, frame_id=self.target_frame),
                              output_fields, np.column_stack([points, intensity]).tolist())
        output.is_dense = True
        self.publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = RegisteredScan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
