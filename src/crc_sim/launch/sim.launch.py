#!/usr/bin/env python3
"""
Launch the complete UEH CRC 2026 simulation:
    Gazebo + 10x6 m track + TurtleBot3 Waffle (camera + LiDAR)

Examples:
    ros2 launch crc_sim sim.launch.py
    ros2 launch crc_sim sim.launch.py rviz:=true
    ros2 launch crc_sim sim.launch.py track_scale:=2.0      # scale the track x2
    ros2 launch crc_sim sim.launch.py camera_pitch:=0.4     # tilt the camera down
    ros2 launch crc_sim sim.launch.py props:=false gui:=false
"""

import os
import tempfile

import xacro
from ament_index_python.packages import (PackageNotFoundError,
                                          get_package_share_directory)
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

BOARD_W = 10.0    # m - real track size
BOARD_H = 6.0     # m

# Start pose: the START lane at the bottom-left, in the MIDDLE of one lane.
# The left edge is a 1400 mm four-lane road, split by a double solid line at
# y = -1.498. Traffic enters heading +X and keeps right, so that is the
# outermost lower lane: y from -1.848 to -2.197, centre y = -2.022.
START_FX = 0.040   # -> world_x = (START_FX - 0.5) * W  => -4.60 m
START_FY = 0.837   # -> world_y = (0.5 - START_FY) * H  => -2.02 m


def launch_setup(context, *_args, **_kwargs):
    pkg = get_package_share_directory('crc_sim')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    scale = float(LaunchConfiguration('track_scale').perform(context))
    props = LaunchConfiguration('props').perform(context)
    w, h = BOARD_W * scale, BOARD_H * scale

    # ---- 1. Generate the world file from xacro ---------------------------
    world_src = os.path.join(pkg, 'worlds', 'crc_track.world.xacro')
    world_doc = xacro.process_file(world_src, mappings={
        'track_w': f'{w:.4f}',
        'track_h': f'{h:.4f}',
        'props': props,
        'skycam': LaunchConfiguration('skycam').perform(context),
        'lights': LaunchConfiguration('lights').perform(context),
        'signs': LaunchConfiguration('signs').perform(context),
        'tunnel_seg': LaunchConfiguration('tunnel_seg').perform(context),
        'mesh_prefix': _mesh_prefix(),
        'use_mesh': LaunchConfiguration('use_mesh').perform(context),
    })
    world_path = os.path.join(tempfile.gettempdir(), 'crc_track.world')
    with open(world_path, 'w') as f:
        f.write(world_doc.toxml())

    # ---- 2. Generate robot_description from xacro ------------------------
    urdf_src = os.path.join(pkg, 'urdf', 'waffle.urdf.xacro')
    robot_desc = xacro.process_file(urdf_src, mappings={
        'use_mesh': LaunchConfiguration('use_mesh').perform(context),
        'camera_pitch': LaunchConfiguration('camera_pitch').perform(context),
        'lidar_range': LaunchConfiguration('lidar_range').perform(context),
        'mesh_prefix': _mesh_prefix(),
        'body_color': LaunchConfiguration('body_color').perform(context),
    }).toxml()

    # ---- 3. Spawn pose ----------------------------------------------------
    sx = LaunchConfiguration('x').perform(context)
    sy = LaunchConfiguration('y').perform(context)
    x = f'{(START_FX - 0.5) * w:.4f}' if sx == 'auto' else sx
    y = f'{(0.5 - START_FY) * h:.4f}' if sy == 'auto' else sy

    gz_args = {'world': world_path, 'verbose': 'true'}

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
            launch_arguments=gz_args.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')),
            condition=IfCondition(LaunchConfiguration('gui')),
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': True, 'robot_description': robot_desc}],
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_waffle',
            output='screen',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'waffle',
                '-x', x, '-y', y, '-z', '0.05',
                '-Y', LaunchConfiguration('yaw').perform(context),
            ],
        ),
        # Drives the lamp colours. Without it the lights sit frozen on
        # whatever they started on and /traffic_lights never publishes, which
        # makes the traffic-light part of the task impossible. It follows the
        # simulation clock, so the phases stay honest on a machine that cannot
        # hold real time.
        Node(
            package='crc_sim',
            executable='traffic_light',
            name='traffic_light',
            output='screen',
            condition=IfCondition(LaunchConfiguration('lights')),
            parameters=[{'use_sim_time': True}],
        ),
        # Walks the pedestrian across the crossing. Conditioned on props,
        # because that is the flag that spawns the pedestrian model itself.
        Node(
            package='crc_sim',
            executable='pedestrian',
            name='pedestrian',
            output='screen',
            condition=IfCondition(LaunchConfiguration('props')),
            parameters=[{
                'use_sim_time': True,
                'dwell': float(LaunchConfiguration('ped_dwell').perform(context)),
                'trigger_range': float(
                    LaunchConfiguration('ped_trigger').perform(context)),
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            condition=IfCondition(LaunchConfiguration('rviz')),
            arguments=['-d', os.path.join(pkg, 'rviz', 'crc.rviz')],
            parameters=[{'use_sim_time': True}],
        ),
    ]


def _mesh_prefix():
    """Absolute file:// path to turtlebot3_description.

    Gazebo rewrites package:// into model:// and then searches
    GAZEBO_MODEL_PATH. Putting the whole ROS share directory on that path works
    but makes the Insert tab of gzclient print an error for every package it
    finds there, several hundred lines of red on every start. An absolute path
    sidesteps the lookup entirely.
    """
    try:
        return 'file://' + get_package_share_directory('turtlebot3_description')
    except PackageNotFoundError:
        return 'package://turtlebot3_description'


def generate_launch_description():
    pkg = get_package_share_directory('crc_sim')
    models_dir = os.path.join(pkg, 'models')

    model_path = os.pathsep.join(
        p for p in (models_dir,
                    os.environ.get('GAZEBO_MODEL_PATH', '')) if p)

    return LaunchDescription([
        # Let Gazebo resolve model://crc_track/...
        # Do NOT set GAZEBO_RESOURCE_PATH here. Overwriting it drops the default
        # /usr/share/gazebo-11, which holds the shader library, and the render
        # engine then fails to start: no camera sensors and gzclient crashes.
        # The track material resolves through model://, so this is enough.
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', model_path),

        # Disable the online model database entirely. Everything still works
        # from local models, and Gazebo can never hang waiting on the network.
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),

        DeclareLaunchArgument('gui', default_value='true',
                              description='Open the Gazebo window'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Open RViz2'),
        DeclareLaunchArgument('props', default_value='true',
                              description='Spawn ramp, tunnel, parked robot and pedestrian'),
        DeclareLaunchArgument('skycam', default_value='true',
                              description='Top-down camera -> /sky_cam/image_raw'),
        DeclareLaunchArgument('lights', default_value='true',
                              description='Spawn the traffic lights'),
        DeclareLaunchArgument('signs', default_value='true',
                              description='Spawn the traffic signs'),
        DeclareLaunchArgument('tunnel_seg', default_value='0.13',
                              description='Tunnel curve wall length: 0.13 = gaps, 0.20 = sealed'),
        DeclareLaunchArgument('ped_dwell', default_value='15.0',
                              description='Seconds the pedestrian waits at each kerb'),
        DeclareLaunchArgument('ped_trigger', default_value='0.0',
                              description='Step out when the robot is this close, '
                                          'in metres; 0 disables it'),
        DeclareLaunchArgument('track_scale', default_value='1.0',
                              description='Track scale factor (1.0 = real 10x6 m)'),
        DeclareLaunchArgument('use_mesh', default_value='true',
                              description='true = use the turtlebot3_description STL meshes'),
        DeclareLaunchArgument('camera_pitch', default_value='0.0',
                              description='Camera downward tilt (rad)'),
        DeclareLaunchArgument('lidar_range', default_value='3.5',
                              description='LiDAR maximum range (m)'),
        DeclareLaunchArgument('body_color', default_value='Gazebo/Orange',
                              description='Robot body colour in Gazebo'),
        DeclareLaunchArgument('x', default_value='auto'),
        DeclareLaunchArgument('y', default_value='auto'),
        DeclareLaunchArgument('yaw', default_value='0.0'),

        OpaqueFunction(function=launch_setup),
    ])
