from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    vel_action_topic = LaunchConfiguration("vel_action_topic")
    request_topic = LaunchConfiguration("request_topic")
    stop_epsilon = LaunchConfiguration("stop_epsilon")

    declare_vel_action_topic = DeclareLaunchArgument(
        "vel_action_topic",
        default_value="/edge/command_action",
        description="Input vel_action topic (EdgeAction)",
    )

    declare_request_topic = DeclareLaunchArgument(
        "request_topic",
        default_value="/api/sport/request",
        description="Output Unitree sport request topic",
    )

    declare_stop_epsilon = DeclareLaunchArgument(
        "stop_epsilon",
        default_value="0.0001",
        description="Absolute threshold to treat velocity as zero",
    )

    executor_node = Node(
        package="edge_rl_runtime",
        executable="real_robot_executor",
        name="real_robot_executor",
        output="screen",
        parameters=[
            {"vel_action_topic": vel_action_topic},
            {"request_topic": request_topic},
            {"stop_epsilon": stop_epsilon},
        ],
    )

    return LaunchDescription([
        declare_vel_action_topic,
        declare_request_topic,
        declare_stop_epsilon,
        
        executor_node,
    ])
