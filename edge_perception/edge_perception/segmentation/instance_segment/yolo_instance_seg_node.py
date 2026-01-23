import random
import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO


class YoloInstanceSegmentNode(Node):
    def __init__(self):
        super().__init__('yolo_instance_segment_node')

        # ---------------- 参数 ----------------
        self.declare_parameter('model_path', 'yolov8n-seg.pt')
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('overlay_topic', '/yolo_seg/overlay')
        self.declare_parameter('mask_alpha', 0.4)

        self.model_path = self.get_parameter('model_path').value
        self.image_topic = self.get_parameter('image_topic').value
        self.overlay_topic = self.get_parameter('overlay_topic').value
        self.mask_alpha = float(self.get_parameter('mask_alpha').value)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device: {self.device}')

        # ---------------- YOLO 模型 ----------------
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        self.get_logger().info(f'Loaded YOLO model: {self.model_path}')

        # ---------------- ROS ----------------
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos
        )
        self.pub = self.create_publisher(Image, self.overlay_topic, 10)

        random.seed(42)

    # ------------------------------------------------
    def image_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge failed: {e}')
            return

        h, w = img.shape[:2]
        overlay = img.copy()

        # ---------------- 推理 ----------------
        with torch.no_grad():
            results = self.model(img, verbose=False)

        res = results[0]
        if res.masks is None:
            return

        masks = res.masks.data  # [N, H, W] (torch)

        for i in range(masks.shape[0]):
            mask = masks[i].cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)

            if mask.shape[:2] != (h, w):
                mask = cv2.resize(
                    mask, (w, h), interpolation=cv2.INTER_NEAREST
                )

            if mask.sum() == 0:
                continue

            # 随机实例颜色
            color = np.array(
                [random.randint(0, 255) for _ in range(3)],
                dtype=np.uint8
            )

            # ---------- 半透明填充 ----------
            colored = np.zeros_like(overlay, dtype=np.uint8)
            colored[mask == 1] = color
            overlay = cv2.addWeighted(
                overlay, 1.0, colored, self.mask_alpha, 0
            )

            # ---------- mask 轮廓 ----------
            contours, _ = cv2.findContours(
                (mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                overlay, contours, -1, color.tolist(), 2
            )

        # ---------------- 发布 ----------------
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(
                overlay, encoding='bgr8'
            )
            overlay_msg.header = msg.header
            self.pub.publish(overlay_msg)
        except Exception as e:
            self.get_logger().warn(f'Publish failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = YoloInstanceSegmentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
