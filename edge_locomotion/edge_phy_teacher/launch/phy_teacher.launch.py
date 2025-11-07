import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from ament_index_python import get_package_share_directory
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():

    # default config path
    default_config_path = os.path.join(
        get_package_share_directory('edge_config'),
        'config',
        'physical_model.yaml'
    )
    
    # phy_teacher config path
    phy_teacher_config_path = os.path.join(
        get_package_share_directory('edge_phy_teacher'),
        'config',
        'phy_teacher.yaml'
    )
    
    # safety set config path
    safety_set_config_path = os.path.join(
        get_package_share_directory('edge_config'),
        'config',
        'safety_set.yaml'
    )

    if not os.path.exists(phy_teacher_config_path):
        raise FileNotFoundError(f"❌ Config file not found: {phy_teacher_config_path}")

    return LaunchDescription([
        # DeclareLaunchArgument(
        #     'update_rate',
        #     default_value='30',
        #     description='Frequency (Hz) of the main control loop'
        # ),
        DeclareLaunchArgument(
            'set_cpu_affinity',
            default_value='1',
            description='Specify which CPU core(s) the node should run on'
        ),
        # DeclareLaunchArgument(
        #     'print_process_time',
        #     default_value='true',
        #     description='Whether to print the process time'
        # ),
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
                default_config_path,
                phy_teacher_config_path,
                safety_set_config_path,
                # {'update_rate': LaunchConfiguration('update_rate')},
                # {'print_process_time': LaunchConfiguration('print_process_time')}
            ],
        )
    ])
