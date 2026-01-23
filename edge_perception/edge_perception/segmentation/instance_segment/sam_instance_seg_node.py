import random
from typing import Optional

import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class SamInstanceSegmentNode(Node):
    def __init__(self):
        super().__init__('sam_instance_segment_node')

        # ---------------- 参数 ----------------
        self.declare_parameter('model_path', 'sam_vit_b_01ec64.pth')
        self.declare_parameter('model_type', 'vit_b')
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('overlay_topic', '/sam/overlay')

        self.declare_parameter('points_per_side', 16)
        self.declare_parameter('pred_iou_thresh', 0.88)
        self.declare_parameter('stability_score_thresh', 0.95)
        self.declare_parameter('min_mask_region_area', 100)

        # ---------------- 读取参数 ----------------
        model_path = self.get_parameter('model_path').value
        model_type = self.get_parameter('model_type').value
        image_topic = self.get_parameter('image_topic').value
        overlay_topic = self.get_parameter('overlay_topic').value

        # ---------------- 设备 ----------------
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device: {self.device}')

        # ---------------- 加载 SAM ----------------
        self.get_logger().info('Loading SAM model...')
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=self.device)

        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=self.get_parameter('points_per_side').value,
            pred_iou_thresh=self.get_parameter('pred_iou_thresh').value,
            stability_score_thresh=self.get_parameter('stability_score_thresh').value,
            min_mask_region_area=self.get_parameter('min_mask_region_area').value
        )

        self.get_logger().info('SAM model loaded')

        # ---------------- ROS 通信 ----------------
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, image_topic, self.image_callback, qos
        )
        self.pub = self.create_publisher(
            Image, overlay_topic, 10
        )

        random.seed(42)
        self.get_logger().info(f'Subscribed to {image_topic}, publishing to {overlay_topic}')

    # ------------------------------------------------
    def image_callback(self, msg: Image):
        try:
            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # ---------------- SAM 推理 ----------------
        try:
            masks = self.mask_generator.generate(image_rgb)
        except Exception as e:
            self.get_logger().error(f'SAM inference failed: {e}')
            return

        overlay = image_rgb.copy()
        np.random.seed(42)

        # ---------------- SAM 风格可视化 ----------------
        for m in masks:
            mask = m["segmentation"]  # bool [H, W]

            color = np.random.randint(0, 255, 3, dtype=np.uint8)

            # 半透明填充
            overlay[mask] = (
                overlay[mask] * 0.6 + color * 0.4
            ).astype(np.uint8)

            # mask 轮廓
            mask_uint8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(
                mask_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                overlay,
                contours,
                -1,
                color.tolist(),
                2
            )

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding='bgr8')
            out_msg.header = msg.header
            self.pub.publish(out_msg)
        except Exception as e:
            self.get_logger().warn(f'Publish failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SamInstanceSegmentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
