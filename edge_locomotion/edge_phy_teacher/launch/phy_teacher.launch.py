from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'update_rate',
            default_value='30',
            description='Frequency (Hz) of the main control loop'
        ),
        DeclareLaunchArgument(
            'set_cpu_affinity',
            default_value='1',
            description='Specify which CPU core(s) the node should run on'
        ),
        DeclareLaunchArgument(
            'print_process_time',
            default_value='false',
            description='Whether to print the process time'
        ),
        Node(
            package='edge_phy_teacher',
            executable='phy_teacher_node',
            name='phy_teacher_node',
            namespace='edge',
            output='screen',
            prefix=PythonExpression([
                "'taskset -c ' + str(",
                LaunchConfiguration('set_cpu_affinity'),
                ")"
            ]),
            parameters=[
                {'update_rate': LaunchConfiguration('update_rate')},
                {'print_process_time': LaunchConfiguration('print_process_time')}
            ],
        )
    ])
