#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression


def generate_launch_description():
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

    use_global_rviz_arg = DeclareLaunchArgument(
        "use_global_rviz", default_value="false", description="Use global RViz (integrated) or not"
    )
    
    robot = LaunchConfiguration("robot")
    namespace = LaunchConfiguration("namespace")
    use_global_rviz = LaunchConfiguration("use_global_rviz")

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
            "use_sim_time": "true",
            "use_rviz": PythonExpression(["'", use_global_rviz, "' == 'false'"])
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
                FindPackageShare("edge_nav_bringup"),
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
        robot_arg,
        namespace_arg,
        use_global_rviz_arg,
        
        # -- Launch Components --
        controller_launch,
        phy_teacher_launch,
        # drl_student_launch,
        trigger_launch,
        virtual_status_indicator_node,
        
        # navigation_launch,
    ])
