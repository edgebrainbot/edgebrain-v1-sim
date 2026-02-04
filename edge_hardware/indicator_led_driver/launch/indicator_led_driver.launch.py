#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Launch file for the Safety LED Driver Node. This node subscribes to the safety status topic 
    and controls two LEDs via Jetson GPIO to visualize safe (green) / unsafe (red) status.
    """

    safety_led_driver_node = Node(
        package='indicator_led_driver',
        executable='safety_led_driver_node',      # ← setup.py entry point
        name='safety_led_driver',
        namespace='edge',                   # Added namespace 'edge'
        output='screen',
        emulate_tty=True,
    )
    
    task_led_driver_node = Node(
        package='indicator_led_driver',
        executable='task_led_driver_node',      # ← setup.py entry point
        name='task_led_driver',
        namespace='edge',                   # Added namespace 'edge'
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        safety_led_driver_node,
        task_led_driver_node
    ])
