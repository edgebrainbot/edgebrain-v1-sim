#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare, FindPackagePrefix


def generate_launch_description():

    # ------------------------------
    # ---    Launch Arguments    ---
    # ------------------------------
    
    log_dir = os.path.join(os.environ.get("ROS_LOG_DIR", os.path.expanduser("~/.ros/log")), "navigation")
    os.makedirs(log_dir, exist_ok=True)

    odom_topic_arg = DeclareLaunchArgument(
        "odom_topic",
        default_value="odom",
        description="Odometry topic to use"
    )
    
    world_name_arg = DeclareLaunchArgument(
        "world_name",
        default_value="garage"
    )

    # -----------------------------
    # Node: visualizationTools
    # -----------------------------
    visualization_tools_node = Node(
        package="visualization_tools",
        executable="visualizationTools",
        name="visualizationTools",
        output="screen",
        parameters=[{
            "metricFile": PathJoinSubstitution([
                log_dir,
                "metrics",
            ]),
            "trajFile": PathJoinSubstitution([
                log_dir,
                "trajectory",
            ]),
            "mapFile": PathJoinSubstitution([
                FindPackageShare("edge_nav_bringup"),
                "mesh",
                LaunchConfiguration("world_name"),
                "map.ply"
            ]),
            "overallMapVoxelSize": 0.1,
            "exploredAreaVoxelSize": 0.2,
            "exploredVolumeVoxelSize": 0.2,
            "transInterval": 0.1,
            "yawInterval": 0.17,
            "overallMapDisplayInterval": 2,
            "exploredAreaDisplayInterval": 1,
        }],
        remappings=[
            ("/state_estimation", LaunchConfiguration("odom_topic")),
        ],
    )

    return LaunchDescription([
        odom_topic_arg,
        world_name_arg,
        
        visualization_tools_node,
        # RealTimePlot disabled (same as XML)
    ])
