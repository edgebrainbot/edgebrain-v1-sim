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

    # ------------------------------
    # ---    Launch Arguments    ---
    # ------------------------------
    namespace_arg = DeclareLaunchArgument(
        "namespace", default_value="edge", description="Global namespace for all nodes"
    )
    robot_arg = DeclareLaunchArgument(
        "robot", default_value="warrior", description="Robot name (e.g. go2, go1, a1)"
    )
    world_name_arg = DeclareLaunchArgument(
        "world_name", default_value="openwild_outdoor.world", description="Gazebo world name"
    )
    use_global_rviz_arg = DeclareLaunchArgument(
        "use_global_rviz", default_value="true", description="Use global RViz (integrated) or not"
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true", description="Use simulation (Gazebo) clock"
    )

    robot = LaunchConfiguration("robot")
    namespace = LaunchConfiguration("namespace")
    world_name = LaunchConfiguration("world_name")
    use_global_rviz = LaunchConfiguration("use_global_rviz")

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
    goal = [20.0, -6.0, 0.0]
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
    
    # -------- Trigger Launch --------
    trigger_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_trigger"),
                "launch",
                "trigger.launch.py"
            ])
        ),
        launch_arguments={
            "teacher_topic_name": "/edge/phy_teacher_action_vel",
        }.items()
    )

    # -------- RL Runtime Env Split Launch --------
    rl_runtime_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_rl_runtime"),
                "launch",
                "rl_runtime.launch.py"
            ])
        ),
        launch_arguments={"task_type": "navigation", "auto_start": "false"}.items()
    )

    # -------- Gazebo Interaction Service Launch --------
    gz_interact_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_gazebo"),
                "launch", "gz_interact.launch.py"
            ])
        ),
        launch_arguments={
            "robot_name": robot,
            "namespace": namespace,
            "world_name": world_name,
            "use_sim_time": str(use_sim_time).lower()
        }.items()
    )

    # -------- Virtual Status Indicator Node --------
    virtual_status_indicator_node = Node(
        package='edge_trigger',
        executable='virtual_status_indicator_node',
        name='virtual_status_indicator',
        # output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    tf_to_map_node = Node(
        package='lidar_to_map',
        executable='registered_scan',
        name='lidar_to_map_node',
        output='screen',
        parameters=[{'use_sim_time': True,
                     'input_topic': '/warrior/lidar/points',
                     'output_topic': '/registered_scan',
                     'target_frame': 'map'}],
    )

    tf_static = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
        parameters=[{'use_sim_time': True}],
    )

    return [
        use_sim_time_arg,
        namespace_arg,
        robot_arg,
        world_name_arg,
        use_global_rviz_arg,
        
        controller_launch,
        joy_node,
        # teleop,
        tf_to_map_node,
        virtual_status_indicator_node,
        trigger_launch,
        rl_runtime_launch,
        gz_interact_launch,
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
