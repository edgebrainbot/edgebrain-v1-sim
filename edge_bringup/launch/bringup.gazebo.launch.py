#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, ExecuteProcess, OpaqueFunction
from launch_ros.actions import Node, SetParameter
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription

def generate_launch_description():

    # ------------------------
    # Launch Arguments
    # ------------------------
    namespace_arg = DeclareLaunchArgument(
        "namespace", default_value="edge", description="Global namespace for all nodes"
    )
    robot_arg = DeclareLaunchArgument(
        "robot", default_value="go2", description="Robot name (e.g. go2, go1, a1)"
    )
    world_name_arg = DeclareLaunchArgument(
        "world_name", default_value="obstacle_course.world", description="Gazebo world name"
    )
    use_global_rviz_arg = DeclareLaunchArgument(
        "use_global_rviz", default_value="false", description="Use global RViz (integrated) or not"
    )
    reset_position_x_arg = DeclareLaunchArgument("reset_position_x", default_value="0.0", description="Reset X")
    reset_position_y_arg = DeclareLaunchArgument("reset_position_y", default_value="0.0", description="Reset Y")
    reset_position_z_arg = DeclareLaunchArgument("reset_position_z", default_value="0.35", description="Reset Z")

    robot = LaunchConfiguration("robot")
    namespace = LaunchConfiguration("namespace")
    world_name = LaunchConfiguration("world_name")
    use_global_rviz = LaunchConfiguration("use_global_rviz")
    reset_position_x = LaunchConfiguration("reset_position_x")
    reset_position_y = LaunchConfiguration("reset_position_y")
    reset_position_z = LaunchConfiguration("reset_position_z")

    set_sim_time = SetParameter(name='use_sim_time', value=True)

    # ------------------------
    # Include Launch Descriptions
    # ------------------------
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_quadruped_controller"),
                "launch", "gazebo.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot,
            "world": world_name,
            "use_sim_time": "true",
            "use_rviz": PythonExpression(["'", use_global_rviz, "' == 'false'"])
        }.items()
    )

    phy_teacher_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_phy_teacher"),
                "launch", "phy_teacher.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot,
            "use_sim_time": "true"
        }.items()
    )

    drl_student_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_drl_student"),
                "launch", "drl_student.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot,
            "use_sim_time": "true"
        }.items()
    )

    trigger_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_trigger"),
                "launch", "trigger.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot,
            "use_sim_time": "true"
        }.items()
    )

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("nav_bringup"),
                "launch", "system_simulation_with_route_planner.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot,
            "world": world_name,
            "use_sim_time": "true",
            "use_rviz": PythonExpression(["'", use_global_rviz, "' == 'false'"])
        }.items()
    )

    # ------------------------
    # Node Definitions
    # ------------------------
    virtual_status_indicator_node = Node(
        package='edge_config',
        executable='virtual_status_indicator_node',
        name='virtual_status_indicator',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    goal = [6.0, -6.0, 0.0]
    publish_goal_point = TimerAction(
        period=1.0,
        actions=[ExecuteProcess(
            cmd=[
                'ros2', 'topic', 'pub', '-r', '100',
                '/goal_point',
                'geometry_msgs/msg/PointStamped',
                f'{{header: {{stamp: {{sec: 0, nanosec: 0}}, frame_id: "map"}}, point: {{x: {goal[0]}, y: {goal[1]}, z: {goal[2]}}}}}'
            ],
            output='screen'
        )]
    )

    gz_reset_handler_node = Node(
        package='edge_bringup', 
        executable='gazebo_reset_handler', 
        name='reset_coordinator_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'world_name': world_name,
            'robot_name': robot,
            'reset_position_x': reset_position_x,
            'reset_position_y': reset_position_y,
            'reset_position_z': reset_position_z,
            'reset_orientation_x': 0.0,
            'reset_orientation_y': 0.0,
            'reset_orientation_z': 0.0,
            'reset_orientation_w': 1.0,
            'service_timeout': 3000,
            'reset_world': False,
            'reset_robot_pose': True,
            'wait_after_reset': 0.5,
        }]
    )

    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('edge_bringup'),
        'rviz',
        'global_view.rviz'
    ])
    
    global_rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='global_rviz',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(use_global_rviz)
    )

    # ------------------------
    # Launch in sequence using OpaqueFunction
    # ------------------------
    def launch_sequence(context, *args, **kwargs):
        actions = []

        # The following launches depend on each other and must be in order
        actions.append(controller_launch)
        actions.append(phy_teacher_launch)
        # actions.append(drl_student_launch)
        actions.append(trigger_launch)
        actions.append(navigation_launch)

        # Other nodes can be launched after the above are set up
        actions.extend([
            virtual_status_indicator_node,
            publish_goal_point,
            gz_reset_handler_node,
            global_rviz_node
        ])
        return actions

    return LaunchDescription([
        # launch arguments
        set_sim_time,
        namespace_arg,
        robot_arg,
        world_name_arg,
        reset_position_x_arg,
        reset_position_y_arg,
        reset_position_z_arg,
        use_global_rviz_arg,

        # Launch in sequence using OpaqueFunction
        OpaqueFunction(function=launch_sequence)
    ])
