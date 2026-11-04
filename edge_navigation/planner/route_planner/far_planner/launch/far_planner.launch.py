import yaml
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import launch_ros

def generate_launch_description():    
    # Far Planner config file path
    config_file = PathJoinSubstitution([
        get_package_share_directory('far_planner'),
        'config',
        [LaunchConfiguration('config'), '.yaml']
    ])
    
    # RViz 
    rviz_config = PathJoinSubstitution([
        get_package_share_directory('far_planner'),
        'rviz',
        [LaunchConfiguration('rviz_config'), '.rviz']
    ])
    
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation time'),
        DeclareLaunchArgument('odom_topic', default_value='odom', description='Odometry topic to use'),
        DeclareLaunchArgument('config', default_value='default', description='Configuration file for RViz and Far Planner'),
        DeclareLaunchArgument('rviz_config', default_value=LaunchConfiguration('config'), description='RViz configuration name'),
        DeclareLaunchArgument('use_rviz', default_value='true', description='Whether to launch RViz'),
        DeclareLaunchArgument('main_run_freq', default_value='5.0', description='Graph and planning update frequency in Hz'),
        LogInfo(msg=['FAR Planner config: ', config_file,
                     '; graph/planning frequency: ', LaunchConfiguration('main_run_freq'), ' Hz']),

        # DeclareLaunchArgument('config', default_value='matterport'),
        Node(
            package='far_planner',
            executable='far_planner',
            name='far_planner',
            output='screen',
            parameters=[
                config_file,
                {'use_sim_time': LaunchConfiguration('use_sim_time'),
                 'main_run_freq': ParameterValue(LaunchConfiguration('main_run_freq'), value_type=float)}
            ],
            remappings=[
                # ('/odom_world', '/state_estimation'),
                ('/odom_world', LaunchConfiguration("odom_topic")),
                ('/terrain_cloud', '/terrain_map_ext'),
                # Dynamic obstacle ray tracing needs the registered raw scan.
                ('/scan_cloud', '/registered_scan'),
                # Terrain classification uses intensity as height above ground.
                ('/terrain_local_cloud', '/terrain_map')
            ]
        ),
        
        Node(
            package='rviz2',
            executable='rviz2',
            name='far_rviz',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            output='screen',
            respawn=False,
            condition=IfCondition(LaunchConfiguration('use_rviz'))
        ),

        # Including another launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([get_package_share_directory('graph_decoder'), '/launch/decoder.launch'])
        )
    ])
