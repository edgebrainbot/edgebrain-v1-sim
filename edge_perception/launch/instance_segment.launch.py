from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('edge_perception')
    default_model_path = os.path.join(pkg_dir, 'models', 'lraspp_mobilenet_w1280_h720_segmentation.onnx')

    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value=default_model_path, description='Path to ONNX instance segmentation model'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image'),
        DeclareLaunchArgument('mask_topic', default_value='/instance_mask'),
        DeclareLaunchArgument('overlay_topic', default_value='/instance_overlay'),
        DeclareLaunchArgument('score_threshold', default_value='0.5'),
        DeclareLaunchArgument('max_detections', default_value='50'),
        DeclareLaunchArgument('input_width', default_value='1280'),
        DeclareLaunchArgument('input_height', default_value='720'),
        Node(
            package='edge_perception',
            executable='instance_segment_node',
            namespace='edge',
            name='instance_segment_node',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'image_topic': LaunchConfiguration('image_topic'),
                'mask_topic': LaunchConfiguration('mask_topic'),
                'overlay_topic': LaunchConfiguration('overlay_topic'),
                'score_threshold': LaunchConfiguration('score_threshold'),
                'max_detections': LaunchConfiguration('max_detections'),
                'input_width': LaunchConfiguration('input_width'),
                'input_height': LaunchConfiguration('input_height'),
            }]
        ),
    ])
