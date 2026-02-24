#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    # -----------------------------
    #      Declare Arguments 
    # -----------------------------
    world_name = DeclareLaunchArgument("world_name", default_value="unity")
    vehicleHeight = DeclareLaunchArgument("vehicleHeight", default_value="0.366")
    sensorOffsetX = DeclareLaunchArgument("sensorOffsetX", default_value="0.0")
    sensorOffsetY = DeclareLaunchArgument("sensorOffsetY", default_value="0.0")
    cameraOffsetZ = DeclareLaunchArgument("cameraOffsetZ", default_value="0.0")
    vehicleX = DeclareLaunchArgument("vehicleX", default_value="0.0")
    vehicleY = DeclareLaunchArgument("vehicleY", default_value="0.0")
    terrainZ = DeclareLaunchArgument("terrainZ", default_value="0.0")
    vehicleYaw = DeclareLaunchArgument("vehicleYaw", default_value="0.0")
    checkTerrainConn = DeclareLaunchArgument("checkTerrainConn", default_value="true")

    # joystick node
    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="ps3_joy",
        output="screen",
        parameters=[
            {"dev": "/dev/input/js0"},
            {"deadzone": 0.12},
            {"autorepeat_rate": 0.0},
            {"use_sim_time": True},
        ],
    )

    # -----------------------------
    # local_planner include
    # -----------------------------
    local_planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        PathJoinSubstitution([
            FindPackageShare("local_planner"),
            "launch",
            "local_planner.launch.py"
        ])),
        launch_arguments={
        "sensorOffsetX": LaunchConfiguration("sensorOffsetX"),
        "sensorOffsetY": LaunchConfiguration("sensorOffsetY"),
        "cameraOffsetZ": LaunchConfiguration("cameraOffsetZ"),
        "goalX": LaunchConfiguration("vehicleX"),
        "goalY": LaunchConfiguration("vehicleY"),
        "odom_topic": LaunchConfiguration("odom_topic"),
        "use_sim_time": "true",
        }.items()
    )

    # -----------------------------
    # terrain_analysis include
    # -----------------------------
    terrain_analysis_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        PathJoinSubstitution([
            FindPackageShare("terrain_analysis"),
            "launch",
            "terrain_analysis.launch.py"
        ])),
        launch_arguments={
            "use_sim_time": "true",
            "odom_topic": LaunchConfiguration("odom_topic"),
        }.items()
    )

    # -----------------------------
    # terrain_analysis_ext include
    # -----------------------------
    terrain_analysis_ext_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        PathJoinSubstitution([
            FindPackageShare("terrain_analysis_ext"),
            "launch",
            "terrain_analysis_ext.launch.py"
        ])),
        launch_arguments={
            "checkTerrainConn": LaunchConfiguration("checkTerrainConn"),
            "odom_topic": LaunchConfiguration("odom_topic"),
            "use_sim_time": "true",
        }.items()
    )

    # -----------------------------
    # vehicle_simulator include
    # -----------------------------
    vehicle_simulator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        PathJoinSubstitution([
            FindPackageShare("vehicle_simulator"),
            "launch",
            "vehicle_simulator.launch.py"
        ])),
        launch_arguments={
        "vehicleHeight": LaunchConfiguration("vehicleHeight"),
        "sensorOffsetX": LaunchConfiguration("sensorOffsetX"),
        "sensorOffsetY": LaunchConfiguration("sensorOffsetY"),
        "vehicleX": LaunchConfiguration("vehicleX"),
        "vehicleY": LaunchConfiguration("vehicleY"),
        "terrainZ": LaunchConfiguration("terrainZ"),
        "vehicleYaw": LaunchConfiguration("vehicleYaw"),
        "use_sim_time": "true",
        }.items()
    )

    # -----------------------------
    # sensor_scan_generation include
    # -----------------------------
    sensor_scan_generation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        PathJoinSubstitution([
            FindPackageShare("sensor_scan_generation"),
            "launch",
            "sensor_scan_generation.launch.py"
        ])),
        launch_arguments={
        "use_sim_time": "true",
        "odom_topic": LaunchConfiguration("odom_topic"),
        }.items()
    )

    # -----------------------------
    # visualization_tools include
    # -----------------------------
    visualization_tools_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        PathJoinSubstitution([
            FindPackageShare("visualization_tools"),
            "launch",
            "visualization_tools.launch.py"
        ])),
        launch_arguments={
        "world_name": LaunchConfiguration("world_name"),
        "odom_topic": LaunchConfiguration("odom_topic"),
        "use_sim_time": "true",
        }.items()
    )

    # -----------------------------
    # ros_tcp_endpoint
    # -----------------------------
    endpoint_node = Node(
        package="ros_tcp_endpoint",
        executable="default_server_endpoint",
        name="endpoint",
        output="screen",
        parameters=[
        {"ROS_IP": "0.0.0.0"},
        {"ROS_TCP_PORT": 10000},
        {"use_sim_time": True},
        ]
    )

    # -----------------------------
    # sim_image_repub (vehicle_simulator)
    # -----------------------------
    sim_image_repub_node = Node(
        package="vehicle_simulator",
        executable="sim_image_repub",
        name="sim_image_repub",
        output="screen",
        parameters=[
        {"camera_in_topic": "/camera/image/compressed"},
        {"camera_raw_out_topic": "/camera/image/raw"},
        {"sem_in_topic": "/camera/semantic_image/compressed"},
        {"sem_raw_out_topic": "/camera/semantic_image/raw"},
        {"depth_in_topic": "/camera/depth/compressed"},
        {"depth_raw_out_topic": "/camera/depth/raw"},
        {"use_sim_time": True}
        ],
    )

    # -----------------------------
    # far_planner include
    # -----------------------------
    far_planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        PathJoinSubstitution([
            FindPackageShare("far_planner"),
            "launch",
            "far_planner.launch.py"
        ])
        ),
        launch_arguments={
            "use_sim_time": "true",
        }.items()
    )

    # -----------------------------
    # Final LaunchDescription
    # -----------------------------
    return LaunchDescription([
        world_name,
        vehicleHeight,
        sensorOffsetX,
        sensorOffsetY,
        cameraOffsetZ,
        vehicleX,
        vehicleY,
        terrainZ,
        vehicleYaw,
        checkTerrainConn,

        joy_node,
        local_planner_launch,
        terrain_analysis_launch,
        terrain_analysis_ext_launch,
        vehicle_simulator_launch,
        sensor_scan_generation_launch,
        visualization_tools_launch,
        endpoint_node,
        sim_image_repub_node,
        far_planner_launch,
    ])
