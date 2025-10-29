from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    # config_path = os.path.join(
    #     os.getenv('AMENT_PREFIX_PATH', '').split(':')[0],
    #     'share', 'edge_drl_student', 'config', 'hyperparams.yaml'
    # )

    return LaunchDescription([
        Node(
            package='edge_drl_student',
            executable='drl_student_node',
            name='drl_student',
            namespace='edge',
            output='screen',
            # parameters=[config_path]
        )
    ])
