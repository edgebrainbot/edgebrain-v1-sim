import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory

def generate_launch_description():
    # Get the path to the parameter file
    physical_model_file = os.path.join(
        get_package_share_directory('edge_config'),
        'config',
        'physical_model.yaml'
    )

    return LaunchDescription([
        Node(
            package='edge_drl_student',
            executable='drl_student_node',
            name='drl_student',
            namespace='edge',
            output='screen',
            parameters=[physical_model_file]
        )
    ])
