#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):

    namespace = LaunchConfiguration("namespace").perform(context)

    # ---- FIX use_sim_time (convert string → bool) ----
    use_sim_time_raw = LaunchConfiguration("use_sim_time").perform(context)
    use_sim_time = (use_sim_time_raw.lower() == "true")

    controller_type = LaunchConfiguration("controller_type").perform(context)

    warrior_description_pkg = FindPackageShare("warrior_description")
    
    warrior_control_pkg = FindPackageShare("warrior_control")
    
    world_name = LaunchConfiguration("world_name").perform(context)

    xacro_file = PathJoinSubstitution([warrior_description_pkg, "urdf", "gzsim.urdf.xacro"])
    robot_description = Command(["xacro ", xacro_file])

    # Choose controller
    if controller_type == "diff_drive_controller":
        controller_file = "diff_drive.gazebo.launch.py"
    else:
        controller_file = "swerve_drive.gazebo.launch.py"

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([warrior_control_pkg, "launch", controller_file])
        ),
        launch_arguments={
            "use_sim_time": str(use_sim_time).lower(),
            "robot_description": robot_description,
            "world_name": world_name,
        }.items()
    )

    # Joy node
    joy_node = Node(
        package="joy",
        executable="joy_node",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    # Teleop node
    teleop = Node(
        package="teleop_twist_joy",
        executable="teleop_node",
        name="teleop_twist_joy",
        parameters=[
            {
            "publish_stamped_twist": True,
            "use_sim_time": use_sim_time,
            "axis_linear.x": 1,
            "axis_linear.y": 0,
            "axis_angular.yaw": 3,
            "scale_linear.x": 1.0,
            "scale_linear.y": 1.0,
            "scale_angular.yaw": 2.0
        }],
        # remappings=[("cmd_vel", f"/{controller_type}/cmd_vel")],
        output="screen",
    )

    # -------- Goal Publisher Node --------
    goal = [6.0, -6.0, 0.0]
    publish_goal_point = TimerAction(
        period=1.0,
        actions=[ExecuteProcess(
            cmd=[
                'ros2', 'topic', 'pub', '-r', '1',
                '/goal_point',
                'geometry_msgs/msg/PointStamped',
                f'{{header: {{stamp: {{sec: 0, nanosec: 0}}, frame_id: "map"}}, point: {{x: {goal[0]}, y: {goal[1]}, z: {goal[2]}}}}}'
            ],
            # output='screen'
        )]
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
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
        parameters=[{'use_sim_time': True}],
    )

    return [
        controller_launch,
        joy_node,
        # teleop,
        # tf_to_map_node,
        tf_static,
        publish_goal_point,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("namespace", default_value="warrior"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("controller_type", default_value="swerve_drive_controller"),
        DeclareLaunchArgument("world_name", default_value="openwild_outdoor.world"),
        # DeclareLaunchArgument("world_name", default_value="competition.world"),
        
        OpaqueFunction(function=launch_setup),
    ])
