#!/usr/bin/env python3
import os
import time
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    ExecuteProcess,
    TimerAction,
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
    
    # ========== NEW: Reset Handler Arguments ==========
    world_name_arg = DeclareLaunchArgument(
        "world_name",
        default_value="obstacle_course",
        description="Gazebo world name for reset"
    )
    
    reset_position_x_arg = DeclareLaunchArgument(
        "reset_position_x",
        default_value="0.0",
        description="Robot reset position X"
    )
    
    reset_position_y_arg = DeclareLaunchArgument(
        "reset_position_y",
        default_value="0.0",
        description="Robot reset position Y"
    )
    
    reset_position_z_arg = DeclareLaunchArgument(
        "reset_position_z",
        default_value="0.35",
        description="Robot reset position Z"
    )


    namespace = LaunchConfiguration("namespace")
    robot = LaunchConfiguration("robot")
    world_name = LaunchConfiguration("world_name")
    reset_position_x = LaunchConfiguration("reset_position_x")
    reset_position_y = LaunchConfiguration("reset_position_y")
    reset_position_z = LaunchConfiguration("reset_position_z")

    # ------------------------
    # Locomotion Control Layer
    # ------------------------
    
    # -------- Body Balance Controller --------
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_quadruped_controller"),
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

    # === Publish goal point (one-shot) ===
    goal = [6.0, -6.0, 0.0]
    publish_goal_point = TimerAction(
        period=1.0, 
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'topic', 'pub', '-r', '100',
                    '/goal_point',
                    'geometry_msgs/msg/PointStamped',
                    f'{{header: {{stamp: {{sec: 0, nanosec: 0}}, frame_id: "map"}}, point: {{x: {goal[0]}, y: {goal[1]}, z: {goal[2]}}}}}'
                ],
                output='screen'
            )
        ]
    )

    # ========== NEW: Reset Handler Node ==========
    gz_reset_handler_node = Node(
        package='edge_bringup', 
        executable='gazebo_reset_handler', 
        name='reset_handler',
        namespace=namespace,
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

    return LaunchDescription([
        # -- Launch Arguments --
        set_sim_time,
        namespace_arg,
        robot_arg,
        world_name_arg,          
        reset_position_x_arg,    
        reset_position_y_arg,   
        reset_position_z_arg,    
        
        # -- Launch Components --
        controller_launch,
        phy_teacher_launch,
        # drl_student_launch,
        trigger_launch,
        virtual_safety_indicator_node,
        
        # navigation_launch,
        publish_goal_point,
        gz_reset_handler_node,
    ])
