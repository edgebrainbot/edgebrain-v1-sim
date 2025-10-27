from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    pkg_edge_trigger = get_package_share_directory('edge_trigger')
    trigger_launch = os.path.join(pkg_edge_trigger, 'launch', 'trigger.launch.py')

    include_trigger = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(trigger_launch),
        launch_arguments={'use_sim_time': 'false'}.items()
    )

    return LaunchDescription([
        include_trigger
    ])
