#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit


def generate_launch_description():
    # ------------------------
    # Declare Launch Arguments
    # ------------------------
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="edge",
        description="Global namespace for all nodes"
    )

    # Robot name argument
    robot_arg = DeclareLaunchArgument(
        "robot",
        default_value="go2",
        description="Robot name (e.g. go2, go1, a1)"
    )
    
    # Real Robot Usage argument
    use_real_robot_arg = DeclareLaunchArgument(
        "use_real_robot",
        default_value="true",
        description="Use real robot hardware interface instead of simulation"
    )
    
    # Network Interface for DDS
    network_interface_arg = DeclareLaunchArgument(
        "network_interface",
        default_value="eno1",
        description="Network interface for DDS communication (e.g., eno1, wlan0, lo)"
    )
    
    # DDS Domain ID
    dds_domain_arg = DeclareLaunchArgument(
        "dds_domain_id",
        default_value="0",
        description="DDS domain ID for communication"
    )

    network_interface = LaunchConfiguration("network_interface")
    dds_domain_id = LaunchConfiguration("dds_domain_id")

    use_real_robot = LaunchConfiguration("use_real_robot")
    namespace = LaunchConfiguration("namespace")
    robot = LaunchConfiguration("robot")

    # ------------------------
    # Hardware layer
    # ------------------------
    led_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("safety_led_driver"),
                "launch",
                "safety_led_driver.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot
        }.items()
    )

    # ------------------------
    # Locomotion Control Layer
    # ------------------------
    
    # -------- Edge Quadruped Controller --------
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_quadruped_controller"),
                "launch",
                "mujoco.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot,
            
            "use_real_robot": use_real_robot,
            "network_interface": network_interface,
            "dds_domain_id": dds_domain_id,

        }.items()
    )
    
    # -------- PHY Teacher --------
    phy_teacher_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_phy_teacher"),
                "launch",
                "phy_teacher.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot
        }.items()
    )
    
    # -------- DRL Student --------
    drl_student_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_drl_student"),
                "launch",
                "drl_student.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot
        }.items()
    )

    # -------- Trigger --------
    trigger_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_trigger"),
                "launch",
                "trigger.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot
        }.items()
    )

    # --------- Navigation Module ---------
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("vehicle_simulator"),
                "launch",
                "system_simulation_with_route_planner.launch.py"
            ])
        ),
        # launch_arguments={
        #     "namespace": namespace,
        #     "robot": robot
        # }.items()
    )

    # === Virtual Status Indicator Node ===
    virtual_status_indicator_node = Node(
        package='edge_config',
        executable='virtual_status_indicator_node',
        name='virtual_status_indicator',
        output='screen',
    )

    return LaunchDescription([
        # -- Launch Arguments --
        namespace_arg,
        robot_arg,
        
        use_real_robot_arg,         
        network_interface_arg,   
        dds_domain_arg,        
    
        # -- Launch Components --
        led_launch,
        controller_launch,
        phy_teacher_launch,
        # drl_student_launch,
        trigger_launch,
        virtual_status_indicator_node,
        
        # navigation_launch,
    ])
