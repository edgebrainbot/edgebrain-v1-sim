from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    request_topic = LaunchConfiguration("request_topic")
    stop_epsilon = LaunchConfiguration("stop_epsilon")

    declare_cmd_vel_topic = DeclareLaunchArgument(
        "cmd_vel_topic",
        default_value="/cmd_vel",
        description="Input cmd_vel topic (TwistStamped)",
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
        package="edge_dummy_env",
        executable="real_robot_executor",
        name="real_robot_executor",
        output="screen",
        parameters=[
            {"cmd_vel_topic": cmd_vel_topic},
            {"request_topic": request_topic},
            {"stop_epsilon": stop_epsilon},
        ],
    )

    return LaunchDescription([
        declare_cmd_vel_topic,
        declare_request_topic,
        declare_stop_epsilon,
        executor_node,
    ])
