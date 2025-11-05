import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    # Config file path
    safety_config_path = os.path.join(
        get_package_share_directory('edge_config'),
        'config', 'safety_set.yaml'
    )
    
    if not os.path.exists(safety_config_path):
        raise FileNotFoundError(f"Config file not found: {safety_config_path}")
    
    # === DRL Student Node ===
    drl_student_node = Node(
        package='edge_drl_student',
        executable='drl_student_node',
        name='drl_student',
        output='screen',
        parameters=[],
    )

    # === PHY Teacher Node ===
    phy_teacher_node = Node(
        package='edge_phy_teacher',
        executable='phy_teacher_node',
        name='phy_teacher',
        output='screen',
        parameters=[safety_config_path],
    )

    # === Trigger Node ===
    trigger_node = Node(
        package='edge_trigger',
        executable='trigger_node',
        name='trigger',
        output='screen',
        parameters=[safety_config_path],
    )

    # === LED Safety Monitor Node ===
    led_safety_monitor_node = Node(
        package='led_safety_monitor',
        executable='safety_monitor_node',
        name='led_safety_monitor',
        output='screen',
        parameters=[],
        emulate_tty=True,
    )


    # === Virtual Safety Indicator Node ===
    virtual_safety_indicator_node = Node(
        package='edge_config',
        executable='virtual_safety_indicator_node',
        name='virtual_safety_indicator',
        output='screen',
    )

    # === Return launch description ===
    return LaunchDescription([
        GroupAction([
            PushRosNamespace('edge'),
            drl_student_node,
            phy_teacher_node,
            trigger_node,
            led_safety_monitor_node,
            virtual_safety_indicator_node,
        ])
    ])
