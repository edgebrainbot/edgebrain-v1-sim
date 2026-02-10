#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import time
from rclpy.node import Node
import Jetson.GPIO as GPIO

# -------------------- GPIO CONFIG --------------------
PIN_LED = [7, 11, 13, 15, 16, 18, 19, 21, 22, 23, 24, 26, 29, 31, 32, 33, 35, 36, 37, 38, 40]
GPIO.setmode(GPIO.BOARD)
GPIO.setup(PIN_LED, GPIO.OUT)


# -------------------- MAIN --------------------
def main(args=None):
    while True:
        GPIO.output(PIN_LED, GPIO.HIGH)  # Red light on
        time.sleep(0.1)
        GPIO.output(PIN_LED, GPIO.LOW)   # Red light off
        time.sleep(0.1)
if __name__ == "__main__":
    main()
