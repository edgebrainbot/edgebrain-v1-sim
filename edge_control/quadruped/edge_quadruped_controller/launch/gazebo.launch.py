import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def launch_setup(context, *args, **kwargs):
    robot_name = context.launch_configurations['robot']
    init_height = context.launch_configurations['height']
    
    pkg_description = f"{robot_name}_description"
    pkg_gazebo = "edge_gazebo"
    
    # Set GAZEBO model path
    model_resource_path = os.path.join(get_package_share_directory(pkg_gazebo), "models")

    os.environ["IGN_GAZEBO_RESOURCE_PATH"] = \
        os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "") + ":" + model_resource_path

    os.environ["GZ_SIM_RESOURCE_PATH"] = \
        os.environ.get("GZ_SIM_RESOURCE_PATH", "") + ":" + model_resource_path
    
    try:
        pkg_path = os.path.join(get_package_share_directory(pkg_description))
    except Exception as e:
        print(f"Error: Could not find package '{pkg_description}'. Please ensure it is installed.")
        print(f"Available packages can be checked with: ros2 pkg list | grep {robot_name}")
        raise e

    xacro_file = os.path.join(pkg_path, 'xacro', 'robot.xacro')
    
    if not os.path.exists(xacro_file):
        print(f"Error: Could not find robot description file: {xacro_file}")
        raise FileNotFoundError(f"Robot description file not found: {xacro_file}")
    
    print(f"Using robot description file: {xacro_file}")
    
    robot_description = xacro.process_file(xacro_file, mappings={'GAZEBO': 'true'}).toxml()

    rviz_config_file = os.path.join(get_package_share_directory("edge_quadruped_controller"), "rviz", "quadruped.rviz")

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_robot_go2',
        output='screen',
        arguments=["-d", rviz_config_file],
        parameters=[{'use_sim_time': True}]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                    'launch',
                                    'gz_sim.launch.py'])]),
            launch_arguments=[
                ('gz_args', [
                    '-r -v 4 ',
                    PathJoinSubstitution([
                        FindPackageShare(pkg_gazebo),
                        'worlds',
                        'simple_indoor.world'
                    ])
                ]),
            ],
    )

    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-name',
                   robot_name, '-allow_renaming', 'true', '-x', '0.', '-y', '0', '-z', init_height],
        parameters=[{'use_sim_time': True}],
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {
                'publish_frequency': 20.0,
                'use_tf_static': True,
                'robot_description': robot_description,
                'ignore_timestamp': True,
                'use_sim_time': True,
            }
        ],
    )

    joint_state_publisher = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster",
                   "--controller-manager", "/controller_manager"],
        parameters=[{'use_sim_time': True}],
    )

    imu_sensor_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["imu_sensor_broadcaster",
                   "--controller-manager", "/controller_manager"],
        parameters=[{'use_sim_time': True}],
    )

    edge_quadruped_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["edge_quadruped_controller", "--controller-manager", "/controller_manager"],
        parameters=[{'use_sim_time': True}],
        remappings=[('odom', 'odom_sim')],
    )

    tf_to_map_node = Node(
        package='lidar_to_map',
        executable='lidar_to_map',
        name='lidar_to_map_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    tf_static = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=["0", "0", "0", "0", "0", "0", "map", "leg_odom"],
        parameters=[{'use_sim_time': True}],
    )


    return [
        tf_static,
        tf_to_map_node,
        
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                        '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
                        '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                        '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
                        '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                        '/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                       ],
            # remappings=[
            #     ('/scan/points', '/registered_scan')
            # ],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

        gazebo,
        robot_state_publisher,
        gz_spawn_entity,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[imu_sensor_broadcaster, joint_state_publisher],
            )
        ),

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_publisher,
                on_exit=[edge_quadruped_controller],
            )
        ),
        
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=edge_quadruped_controller,
                on_exit=[rviz],
            )
        ),
    ]


def generate_launch_description():
    robot_name = DeclareLaunchArgument(
        'robot',
        default_value='go2',
        description='Robot name (will automatically append _description to find the description package)'
    )

    height = DeclareLaunchArgument(
        'height',
        default_value='0.5',
        description='Init height in simulation'
    )

    return LaunchDescription([
        robot_name,
        height,
        OpaqueFunction(function=launch_setup),
    ])
