import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 0. Khai báo Argument 'lights' (mặc định là 'true')
    lights_arg = DeclareLaunchArgument(
        'lights',
        default_value='true',
        description='Có bật node đèn giao thông hay không (true/false)'
    )

    # 1. Node Đèn giao thông

    # 2. Node Nhận diện Đèn/Biển báo
    traffic_perception_node = Node(
        package='crc_sim',
        executable='den',
        name='traffic_perception_tester',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # 3. Node Điều khiển xe
    starter_node = Node(
        package='crc_sim',
        executable='start',
        name='crc_start',
        output='screen',
        parameters=[{'use_sim_time': True}] 
    )

    return LaunchDescription([
        lights_arg,
        traffic_perception_node,
        starter_node,
    ])