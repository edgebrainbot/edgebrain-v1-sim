import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class InstanceSegmentNode(Node):
    def __init__(self) -> None:
        super().__init__('instance_segment_node')
        self.declare_parameter('model_path', '')
        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('mask_topic', '/instance_mask')
        self.declare_parameter('overlay_topic', '/instance_overlay')
        self.declare_parameter('score_threshold', 0.5)
        self.declare_parameter('max_detections', 50)
        self.declare_parameter('input_width', 1280)
        self.declare_parameter('input_height', 720)

        self.model_path: str = self.get_parameter('model_path').get_parameter_value().string_value
        self.score_threshold: float = self.get_parameter('score_threshold').get_parameter_value().double_value
        self.max_detections: int = self.get_parameter('max_detections').get_parameter_value().integer_value
        self.input_width: int = self.get_parameter('input_width').get_parameter_value().integer_value
        self.input_height: int = self.get_parameter('input_height').get_parameter_value().integer_value

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.mask_topic = self.get_parameter('mask_topic').get_parameter_value().string_value
        self.overlay_topic = self.get_parameter('overlay_topic').get_parameter_value().string_value

        qos_reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST)

        self.mask_pub = self.create_publisher(Image, self.mask_topic, 10)
        self.overlay_pub = self.create_publisher(Image, self.overlay_topic, 10)
        self.bridge = CvBridge()

        self.session: Optional['ort.InferenceSession'] = None
        self.input_name: Optional[str] = None
        self.output_names: List[str] = []
        self._init_session()

        self.palette = self._build_palette(80)

        self.image_sub = self.create_subscription(Image, image_topic, self._image_callback, qos_reliable)
        self.get_logger().info(f'Instance seg subscribed to: {image_topic}')

    def _init_session(self) -> None:
        if self.model_path == '':
            self.get_logger().error('model_path is empty; set a valid ONNX model path.')
            return
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.get_logger().info(f'Loaded ONNX model: {self.model_path}')
        except Exception as exc:
            self.get_logger().error(f'Failed to load model {self.model_path}: {exc}')
            self.session = None

    @staticmethod
    def _build_palette(k: int) -> np.ndarray:
        random.seed(42)
        colors = [(128, 128, 128)]
        for _ in range(k):
            colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        return np.array(colors, dtype=np.uint8)

    def _preprocess(self, cv_image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w, _ = cv_image.shape
        resized = cv2.resize(cv_image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, (h, w)

    def _parse_outputs(self, outputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        names = self.output_names
        data: Dict[str, np.ndarray] = {}
        for name, arr in zip(names, outputs):
            data[name] = arr
        # Fallback for common ordering if names are missing
        if 'boxes' not in data and len(outputs) >= 4:
            data.setdefault('boxes', outputs[0])
            data.setdefault('labels', outputs[1])
            data.setdefault('scores', outputs[2])
            data.setdefault('masks', outputs[3])
        return data

    def _image_callback(self, image_msg: Image) -> None:
        if self.session is None:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'cv_bridge failed: {exc}', throttle_duration_sec=3.0)
            return

        tensor, (orig_h, orig_w) = self._preprocess(cv_image)
        try:
            raw_outputs = self.session.run(None, {self.input_name: tensor})
        except Exception as exc:
            self.get_logger().warn(f'Inference failed: {exc}', throttle_duration_sec=3.0)
            return

        parsed = self._parse_outputs(raw_outputs)
        boxes = parsed.get('boxes')
        labels = parsed.get('labels')
        scores = parsed.get('scores')
        masks = parsed.get('masks')

        if boxes is None or labels is None or scores is None or masks is None:
            self.get_logger().warn('ONNX outputs missing boxes/labels/scores/masks', throttle_duration_sec=3.0)
            return

        keep = scores.flatten() > self.score_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        masks = masks[keep]
        scores = scores[keep]

        if boxes.shape[0] == 0:
            self.get_logger().info('No instances above threshold', throttle_duration_sec=3.0)
            return

        if boxes.shape[0] > self.max_detections:
            boxes = boxes[: self.max_detections]
            labels = labels[: self.max_detections]
            masks = masks[: self.max_detections]
            scores = scores[: self.max_detections]

        # Resize masks to original size
        resized_masks = []
        for m in masks:
            # m shape could be (1, h, w) or (h, w)
            if m.ndim == 3:
                m = m[0]
            m_resized = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            resized_masks.append(m_resized)
        masks_resized = np.stack(resized_masks, axis=0)

        color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        overlay = cv_image.copy()

        for idx in range(boxes.shape[0]):
            mask_bin = masks_resized[idx] > 0.5
            if mask_bin.sum() == 0:
                continue
            color = self.palette[(int(labels[idx]) + idx) % self.palette.shape[0]].tolist()
            color_mask[mask_bin] = color
            cv2.rectangle(overlay, (int(boxes[idx][0]), int(boxes[idx][1])), (int(boxes[idx][2]), int(boxes[idx][3])), color, 2)
            cv2.putText(overlay,
                        f'id{idx} cls{int(labels[idx])} {scores[idx]:.2f}',
                        (int(boxes[idx][0]), int(boxes[idx][1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            # Blend mask onto overlay
            overlay = cv2.addWeighted(overlay, 1.0, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR), 0.4, 0)

        try:
            mask_msg = self.bridge.cv2_to_imgmsg(color_mask, encoding='rgb8')
            mask_msg.header = image_msg.header
            self.mask_pub.publish(mask_msg)
        except Exception as exc:
            self.get_logger().warn(f'Publish mask failed: {exc}', throttle_duration_sec=3.0)

        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            overlay_msg.header = image_msg.header
            self.overlay_pub.publish(overlay_msg)
        except Exception as exc:
            self.get_logger().warn(f'Publish overlay failed: {exc}', throttle_duration_sec=3.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InstanceSegmentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
