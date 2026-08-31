#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    TimerAction,
    ExecuteProcess,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_file_dir = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'), 'launch'
    )
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # ── Launch arguments ──────────────────────────────────────────────────────
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose       = LaunchConfiguration('x_pose',       default='2.5')
    y_pose       = LaunchConfiguration('y_pose',       default='-2.0')
    yaw_pose     = LaunchConfiguration('yaw_pose',     default='1.5708')  # 0° (doğu)

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation (Gazebo) clock'
    )
    declare_x_pose = DeclareLaunchArgument(
        'x_pose', default_value='2.5',
        description='Initial X position of the robot'
    )
    declare_y_pose = DeclareLaunchArgument(
        'y_pose', default_value='-2.0',
        description='Initial Y position of the robot'
    )
    declare_yaw_pose = DeclareLaunchArgument(
        'yaw_pose', default_value='1.5708',
        description='Initial yaw of the robot in radians (CCW positive)'
    )

    # ── World file ────────────────────────────────────────────────────────────
    world = os.path.join(
        get_package_share_directory('my_config'),
        'worlds',
        'static_world.world'
    )

    sdf = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models', 'turtlebot3_burger', 'model.sdf'
    )

    # ── 0. Önceki session kalıntısını temizle ─────────────────────────────────
    cleanup_entity = ExecuteProcess(
        cmd=[
            'bash', '-c',
            'ros2 service call /delete_entity '
            'gazebo_msgs/srv/DeleteEntity '
            '"{name: burger}" || true'
        ],
        output='screen'
    )

    # ── 1. Gazebo server ──────────────────────────────────────────────────────
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    # ── 2. Gazebo client / GUI ────────────────────────────────────────────────
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    # ── 3. Robot state publisher ──────────────────────────────────────────────
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

   	 # ── 4. Spawn — spawn_turtlebot3.launch.py yaw desteklemediği için
    #    spawn_entity.py'yi doğrudan çağırıyoruz; -Y ile yaw verebiliyoruz.
    spawn_turtlebot_cmd = TimerAction(
        period=6.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                    '-entity', 'burger',
                    '-file',   sdf,
                    '-x', '2.5',
                    '-y', '-2.0',
                    '-z', '0.01',
                    '-Y', '1.5708',   # 0° — doğuya bak
                ],
                output='screen'
            )
        ]
    )
    
    # ── LaunchDescription ─────────────────────────────────────────────────────
    ld = LaunchDescription()

    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_x_pose)
    ld.add_action(declare_y_pose)
    ld.add_action(declare_yaw_pose)

    ld.add_action(cleanup_entity)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)  # 6 sn gecikmeli

    return ld
