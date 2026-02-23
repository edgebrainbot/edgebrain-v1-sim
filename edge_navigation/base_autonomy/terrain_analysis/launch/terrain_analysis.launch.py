#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    terrain_analysis_node = Node(
        package="terrain_analysis",
        executable="terrainAnalysis",
        name="terrainAnalysis",
        output="screen",
        parameters=[{
            "scanVoxelSize": 0.05,
            "decayTime": 2.0,
            "noDecayDis": 0.0,
            "clearingDis": 8.0,
            "useSorting": True,
            "quantileZ": 0.25,
            "considerDrop": False,
            "limitGroundLift": False,
            "maxGroundLift": 0.15,
            "clearDyObs": False,
            "minDyObsDis": 0.3,
            "minDyObsAngle": 0.0,
            "minDyObsRelZ": -0.3,
            "absDyObsRelZThre": 0.2,
            "minDyObsVFOV": -16.0,
            "maxDyObsVFOV": 16.0,
            "minDyObsPointNum": 1,
            "noDataObstacle": True,
            "noDataBlockSkipNum": 0,
            "minBlockPointNum": 10,
            "maxElevBelowVeh": -0.6,
            "noDataAreaMinX": 0.3,
            "noDataAreaMaxX": 1.8,
            "noDataAreaMinY": -0.9,
            "noDataAreaMaxY": 0.9,
            "vehicleHeight": 1.5,
            "voxelPointUpdateThre": 100,
            "voxelTimeUpdateThre": 2.0,
            "minRelZ": -1.5,
            "maxRelZ": 0.5,
            "disRatioZ": 0.2,
            # 'use_sim_time': True,
        }],
        remappings=[
            ("/state_estimation", "/odom"),
        ],
    )

    return LaunchDescription([
        terrain_analysis_node
    ])
