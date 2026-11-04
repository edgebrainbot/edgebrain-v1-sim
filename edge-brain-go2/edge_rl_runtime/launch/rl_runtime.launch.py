import os
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python import get_package_share_directory


def generate_launch_description():
    topic_namespace = LaunchConfiguration("topic_namespace")

    config_file = os.path.join(
        get_package_share_directory("edge_rl_runtime"), 
        'config', 
        'rl_runtime.yaml'
    )

    rl_runtime_node = Node(
        package="edge_rl_runtime",
        executable="rl_runtime_node",
        name="rl_runtime_node",
        namespace=topic_namespace,
        output="screen",
        parameters=[
            config_file,
            {"task_type": LaunchConfiguration("task_type"),
             "auto_start": ParameterValue(LaunchConfiguration("auto_start"), value_type=bool)},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "auto_start", default_value="false",
            description="Automatically start episode processing in simulation",
        ),
        DeclareLaunchArgument(
            "task_type", default_value="end2end",
            description="Runtime task: navigation, locomotion or end2end",
        ),
        DeclareLaunchArgument(
            "go2_state_topic",
            default_value="/edge/go2/high_state_vec12",
            description="ROS topic used to relay GO2 high state Vec12",
        ),
        DeclareLaunchArgument(
            "topic_namespace",
            default_value="edge",
            description="Namespace prefix used by rl_runtime_node for subscribing to GO2 state and publishing cmd_vel and transition",
        ),
        
        # Nodes
        rl_runtime_node,
    ])
