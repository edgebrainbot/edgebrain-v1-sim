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
from launch_ros.actions import SetParameter

def generate_launch_description():
    
    set_sim_time = SetParameter(name='use_sim_time', value=True)

    # ------------------------
    # Declare Launch Arguments
    # ------------------------
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="edge",
        description="Global namespace for all nodes"
    )

    robot_arg = DeclareLaunchArgument(
        "robot",
        default_value="go2",
        description="Robot name (e.g. go2, go1, a1)"
    )

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
            "robot": robot,
            "use_sim_time": "true"
        }.items()
    )

    # ------------------------
    # Locomotion Control Layer
    # ------------------------
    
    # -------- Body Balance Controller --------
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("body_balance_controller"),
                "launch",
                "gazebo.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot,
            "use_sim_time": "true"
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
            "robot": robot,
            "use_sim_time": "true"
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
            "robot": robot,
            "use_sim_time": "true"
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
            "robot": robot,
            "use_sim_time": "true"
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

    # === Virtual Safety Indicator Node ===
    virtual_safety_indicator_node = Node(
        package='edge_config',
        executable='virtual_safety_indicator_node',
        name='virtual_safety_indicator',
        output='screen',
        parameters=[{
            'use_sim_time': True,
        }]
    )

    return LaunchDescription([
        # -- Launch Arguments --
        set_sim_time,
        namespace_arg,
        robot_arg,
        
        # -- Launch Components --
        led_launch,
        controller_launch,
        phy_teacher_launch,
        # drl_student_launch,
        trigger_launch,
        virtual_safety_indicator_node,
        
        # navigation_launch,
    ])
