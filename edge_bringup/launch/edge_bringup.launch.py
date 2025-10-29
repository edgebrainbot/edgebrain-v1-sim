from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
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
        parameters=[],
    )

    # === Trigger Node ===
    trigger_node = Node(
        package='edge_trigger',
        executable='trigger_node',
        name='trigger',
        output='screen',
        parameters=[],
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


    # === Return launch description ===
    return LaunchDescription([
        GroupAction([
            PushRosNamespace('edge'),
            drl_student_node,
            phy_teacher_node,
            trigger_node,
            led_safety_monitor_node,
        ])
    ])
