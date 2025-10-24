#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Launch file for the Safety Monitor Node.
    This node monitors robot safety states (roll, pitch, IMU, body height)
    and controls an LED indicator via Jetson GPIO to visualize patch/safety status.
    """

    safety_monitor_node = Node(
        package='led_safety_monitor',              
        executable='safety_monitor_node',      # ← setup.py entry point
        name='led_safety_monitor',
        namespace='edge',                   # Added namespace 'edge'
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([safety_monitor_node])
