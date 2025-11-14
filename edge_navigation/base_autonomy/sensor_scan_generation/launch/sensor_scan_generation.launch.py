#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    sensor_scan_gen_node = Node(
        package="sensor_scan_generation",
        executable="sensorScanGeneration",
        name="sensorScanGeneration",
        output="screen"
    )

    return LaunchDescription([
        sensor_scan_gen_node
    ])
