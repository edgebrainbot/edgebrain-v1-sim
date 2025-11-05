import os

import yaml
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Config file path
    safety_config_path = os.path.join(
        get_package_share_directory('edge_config'),
        'config', 'safety_set.yaml'
    )
    
    if not os.path.exists(safety_config_path):
        raise FileNotFoundError(f"Config file not found: {safety_config_path}")

    # Declare launch arguments
    update_rate_arg = DeclareLaunchArgument(
        'update_rate',
        default_value='50.0',
        description='Frequency (Hz) of the trigger loop'
    )

    # Create node
    trigger_node = Node(
        package='edge_trigger',
        executable='trigger_node',
        namespace='edge',
        name='trigger',
        output='screen',
        parameters=[
            safety_config_path,
            # {'update_rate': LaunchConfiguration('update_rate')}
        ]
    )
    
    return LaunchDescription([
        update_rate_arg,  
        trigger_node,
    ])