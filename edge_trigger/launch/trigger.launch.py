#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    trigger_config_path = os.path.join(
        get_package_share_directory('edge_trigger'),
        'config',
        'trigger.yaml'
    )

    update_rate_arg = DeclareLaunchArgument(
        'update_rate',
        default_value='500.0',
        description='Trigger update rate (Hz)'
    )

    print_processing_arg = DeclareLaunchArgument(
        'print_processing',
        default_value='false',
        description='Print processing logs'
    )
    
    start_with_teacher_arg = DeclareLaunchArgument(
        'start_with_teacher',
        default_value='true',
        description='Start in teacher mode if true, else start in student mode. \
            Starting mode is important for safety.'
    )
    
    action_source_arg = DeclareLaunchArgument(
        'action_source',
        default_value='-1',
        description='Action source: -1 -- none, 0 -- student, 1 -- teacher'
    )

    trigger_node = Node(
        package='edge_trigger',
        executable='trigger',
        namespace='edge',
        name='trigger',
        output='screen',
        parameters=[
            trigger_config_path,        # Load default config from YAML
            {
                # Override parameters with launch arguments
                'update_rate': LaunchConfiguration('update_rate'),
                'print_processing': LaunchConfiguration('print_processing'),
                'start_with_teacher': LaunchConfiguration('start_with_teacher'),
                'action_source': LaunchConfiguration('action_source'),
            }
        ]
    )

    return LaunchDescription([
        update_rate_arg,
        print_processing_arg,
        start_with_teacher_arg,
        action_source_arg,
        
        trigger_node,
    ])
