#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import rclpy
from std_msgs.msg import Bool
from rclpy.node import Node
import Jetson.GPIO as GPIO

# -------------------- GPIO CONFIG --------------------
PIN_LED = 17
GPIO.setmode(GPIO.BOARD)
GPIO.setup(PIN_LED, GPIO.OUT)

def set_led(red_on: bool):
    GPIO.output(PIN_LED, GPIO.HIGH if red_on else GPIO.LOW)


# -------------------- Task LED Driver Node --------------------
class TaskLedDriverNode(Node):
    def __init__(self):
        super().__init__("task_led_driver")
        self.get_logger().info("🟢 Task LED Driver Node started.")

        # Subscribe to task_done status topic
        self.sub_safe_ = self.create_subscription(
            msg_type = Bool,  # import std_msgs.msg.Bool
            topic = "is_done",
            callback = self.task_callback,
            qos_profile = 10,
        )

    def task_callback(self, msg: Bool):
        """callback when receiving is_done message"""
        if msg.data:
            set_led(False)  # task undone → no light on
            self.get_logger().info("✅ System is safe.")
        else:
            set_led(True)   # task done → blue light on
            self.get_logger().warn("⚠️ System is unsafe!")

    def destroy_node(self):
        GPIO.cleanup()
        super().destroy_node()
        self.get_logger().info("💡 GPIO cleaned up and node shut down.")
        

# -------------------- MAIN --------------------
def main(args=None):
    rclpy.init(args=args)
    node = TaskLedDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
