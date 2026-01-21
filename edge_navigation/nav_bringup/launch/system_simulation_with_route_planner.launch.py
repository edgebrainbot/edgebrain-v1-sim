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
    use_rviz = DeclareLaunchArgument("use_rviz", default_value="true")
    world_name = DeclareLaunchArgument("world_name", default_value="unity")
    vehicleHeight = DeclareLaunchArgument("vehicleHeight", default_value="0.30")
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

    # --------------
    # local_planner 
    # --------------
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
        "use_sim_time": "true",
        }.items()
    )

    # -----------------
    # terrain_analysis
    # -----------------
    terrain_analysis_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        PathJoinSubstitution([
            FindPackageShare("terrain_analysis"),
            "launch",
            "terrain_analysis.launch.py"
        ])),
        launch_arguments={
            "use_sim_time": "true",
        }.items()
    )

    # ---------------------
    # terrain_analysis_ext 
    # ---------------------
    terrain_analysis_ext_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        PathJoinSubstitution([
            FindPackageShare("terrain_analysis_ext"),
            "launch",
            "terrain_analysis_ext.launch.py"
        ])),
        launch_arguments={
            "checkTerrainConn": LaunchConfiguration("checkTerrainConn"),
            "use_sim_time": "true",
        }.items()
    )

    # -----------------------
    # sensor_scan_generation
    # -----------------------
    sensor_scan_generation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        PathJoinSubstitution([
            FindPackageShare("sensor_scan_generation"),
            "launch",
            "sensor_scan_generation.launch.py"
        ])),
        launch_arguments={
        "use_sim_time": "true",
        }.items()
    )

    # --------------------
    # visualization_tools
    # --------------------
    visualization_tools_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
        PathJoinSubstitution([
            FindPackageShare("visualization_tools"),
            "launch",
            "visualization_tools.launch.py"
        ])),
        launch_arguments={
        "world_name": LaunchConfiguration("world_name"),
        "use_sim_time": "true",
        }.items()
    )

    # ------------
    # far_planner
    # ------------
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
            "use_rviz": LaunchConfiguration("use_rviz"),
        }.items()
    )

    # -----------------------------
    # Final LaunchDescription
    # -----------------------------
    return LaunchDescription([
        use_rviz,
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
        sensor_scan_generation_launch,
        visualization_tools_launch,

        far_planner_launch,
    ])
