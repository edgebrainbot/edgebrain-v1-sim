#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import rclpy
from rclpy.node import Node
import Jetson.GPIO as GPIO

# -------------------- GPIO CONFIG --------------------
PIN_LED = 13
GPIO.setmode(GPIO.BOARD)
GPIO.setup(PIN_LED, GPIO.OUT)

def set_led(red_on: bool):
    GPIO.output(PIN_LED, GPIO.HIGH if red_on else GPIO.LOW)

# -------------------- Safety LED Driver Node --------------------
class SafetyLedDriverNode(Node):
    def __init__(self):
        super().__init__("safety_led_driver")
        self.get_logger().info("🟢 Safety LED Driver Node started.")

        # Subscribe to safety status topic
        self.sub_safe_ = self.create_subscription(
            msg_type = __import__('std_msgs.msg').msg.Bool,  # import std_msgs.msg.Bool
            topic = "is_safe",
            callback = self.safe_callback,
            qos_profile = 10,
        )

    def safe_callback(self, msg):
        """callback when receiving is_safe message"""
        if msg.data:
            set_led(False)  # safe → green light on
            self.get_logger().info("✅ System is safe.")
        else:
            set_led(True)   # unsafe → red light on
            self.get_logger().warn("⚠️ System is unsafe!")


    def destroy_node(self):
        GPIO.cleanup()
        super().destroy_node()
        self.get_logger().info("💡 GPIO cleaned up and node shut down.")

# -------------------- MAIN --------------------
def main(args=None):
    rclpy.init(args=args)
    node = SafetyLedDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
