from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    go2_network_interface = LaunchConfiguration("go2_network_interface")
    go2_state_topic = LaunchConfiguration("go2_state_topic")
    go2_publish_hz = LaunchConfiguration("go2_publish_hz")
    topic_namespace = LaunchConfiguration("topic_namespace")

    return LaunchDescription([
        DeclareLaunchArgument(
            "go2_network_interface",
            default_value="eno1",
            description="Network interface used by Unitree SDK",
        ),
        DeclareLaunchArgument(
            "go2_state_topic",
            default_value="/edge/go2/high_state_vec12",
            description="ROS topic used to relay GO2 high state Vec12",
        ),
        DeclareLaunchArgument(
            "go2_publish_hz",
            default_value="100.0",
            description="Bridge publish rate in Hz",
        ),
        DeclareLaunchArgument(
            "topic_namespace",
            default_value="edge",
            description="Namespace prefix used by rl_runtime_node for subscribing to GO2 state and publishing cmd_vel and transition",
        ),
        Node(
            package="edge_rl_runtime",
            executable="go2_high_state_bridge",
            name="go2_high_state_bridge",
            output="screen",
            parameters=[{
                "go2_network_interface": go2_network_interface,
                "state_topic": go2_state_topic,
                "publish_hz": go2_publish_hz,
            }],
        ),
        Node(
            package="edge_rl_runtime",
            executable="rl_runtime_node",
            name="rl_runtime_node",
            output="screen",
            parameters=[{
                "topic_namespace": topic_namespace,
                "go2_state_topic": go2_state_topic,
            }],
        ),
    ])
