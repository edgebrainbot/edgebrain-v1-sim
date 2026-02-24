import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from unitree_go.msg import Go2FrontVideoData
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import subprocess
import numpy as np
import cv2
import threading


class Go2CameraBridge(Node):

    def __init__(self):
        super().__init__('go2_camera_bridge')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub = self.create_subscription(
            Go2FrontVideoData,
            '/frontvideostream',
            self.callback,
            qos
        )

        self.pub = self.create_publisher(
            Image,
            '/go2/image_raw',
            10
        )

        self.bridge = CvBridge()

        # 启动 ffmpeg：H264 → MJPEG pipe
        self.proc = subprocess.Popen(
            [
                'ffmpeg',
                '-loglevel', 'quiet',
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-i', 'pipe:0',
                '-f', 'image2pipe',
                '-vcodec', 'mjpeg',
                '-'
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

        self.read_thread = threading.Thread(target=self.read_frames)
        self.read_thread.daemon = True
        self.read_thread.start()

        self.get_logger().info("GO2 camera bridge started.")

    def callback(self, msg):
        if not msg.video720p:
            return

        try:
            self.proc.stdin.write(bytes(msg.video720p))
            self.proc.stdin.flush()
        except Exception as e:
            self.get_logger().warn(f"Write error: {e}")

    def read_frames(self):
        buffer = b''

        while rclpy.ok():
            try:
                chunk = self.proc.stdout.read(4096)
                if not chunk:
                    continue

                buffer += chunk

                # 查找 JPEG 帧
                while True:
                    start = buffer.find(b'\xff\xd8')
                    end = buffer.find(b'\xff\xd9')

                    if start != -1 and end != -1 and end > start:
                        jpg = buffer[start:end+2]
                        buffer = buffer[end+2:]

                        np_arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if frame is not None:
                            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                            self.pub.publish(ros_image)
                    else:
                        break

            except Exception as e:
                self.get_logger().warn(f"Read error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = Go2CameraBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()