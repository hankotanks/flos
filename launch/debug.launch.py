# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import OpaqueFunction

PACKAGE_NAME = "forest_segmentation"

def forest_segmentation_setup(context, *args, **kwargs):
    path_to_bag = LaunchConfiguration("path_to_bag").perform(context) # REQUIRED: path to the rosbag
    voxel_size = LaunchConfiguration("voxel_size").perform(context)
    incoming_points_topic = LaunchConfiguration("incoming_points_topic").perform(context)

    segmentation_node_offline_parameters = {
        'use_sim_time': True,
        'path_to_bag': path_to_bag
    }
    if len(incoming_points_topic) > 0:
        segmentation_node_offline_parameters['incoming_points_topic'] = incoming_points_topic
    voxel_size = float(voxel_size)
    if voxel_size > 0:
        segmentation_node_offline_parameters['voxel_size'] = voxel_size

    odometry_node_parameters = {
        'use_sim_time': True,
        'debug': True,
    }
    if voxel_size > 0:
        odometry_node_parameters['voxel_size'] = voxel_size * 1.5

    return [
        Node(
            package=PACKAGE_NAME,
            executable='segmentation_node_offline',
            parameters=[segmentation_node_offline_parameters]
        ),
        Node(
            package=PACKAGE_NAME,
            executable="odometry_node",
            parameters=[odometry_node_parameters]
        ),
    ]

def generate_launch_description():
    visualize = LaunchConfiguration("visualize") # OPTIONAL: run rviz alongside the nodes

    return LaunchDescription([
        DeclareLaunchArgument('visualize', default_value="true"),
        DeclareLaunchArgument('path_to_bag'),
        DeclareLaunchArgument('voxel_size', default_value="1"),
        DeclareLaunchArgument('incoming_points_topic', default_value=""),
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            condition=IfCondition(visualize),
            arguments=["-d", PathJoinSubstitution([FindPackageShare(PACKAGE_NAME), "rviz", "debug.rviz"])],
            parameters=[{ 'use_sim_time': True }]
        ),
        OpaqueFunction(function=forest_segmentation_setup)
    ])
