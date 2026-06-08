from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import OpaqueFunction

PACKAGE_NAME = "forest_segmentation"

def param_str(context, params, name):
    assert isinstance(params, dict) or isinstance(params, list)
    if isinstance(params, dict): params = [params]
    value = LaunchConfiguration(name).perform(context)
    if value is None: return None
    if len(value) > 0:
        for p in params: p[name] = value
        return value
    return None

def param_bool(context, params, name):
    assert isinstance(params, dict) or isinstance(params, list)
    if isinstance(params, dict): params = [params]
    value = LaunchConfiguration(name).perform(context)
    if value is None: return None
    if len(value) > 0:
        assert value in ["true", "false"]
        for p in params: p[name] = True if value == "true" else False
        return value
    return None

def param_int(context, params, name):
    assert isinstance(params, dict) or isinstance(params, list)
    if isinstance(params, dict): params = [params]
    value = LaunchConfiguration(name).perform(context)
    if value is None: return None
    if len(value) > 0:
        value = int(value)
        for p in params: p[name] = value
        return value
    return None

def param_float(context, params, name):
    assert isinstance(params, dict) or isinstance(params, list)
    if isinstance(params, dict): params = [params]
    value = LaunchConfiguration(name).perform(context)
    if value is None: return None
    if len(value) > 0:
        value = float(value)
        for p in params: p[name] = value
        return value
    return None

def setup(context, *args, **kwargs):
    p_seg = {
        'use_sim_time': True
    }

    p_odom = {
        'use_sim_time': True,
        'debug': True,
    }
    
    # shared parameters 
    assert (p := param_float(context, [p_seg, p_odom], "voxel_size")) is None or p > 0.0
    assert (p := param_float(context, [p_seg, p_odom], "max_distance")) is None or p > 0.0
    # segmentation parameters
    assert param_str(context, p_seg, "path_to_bag") is not None
    param_str(context, p_seg, "path_to_model")
    param_str(context, p_seg, "incoming_points_topic")
    param_bool(context, p_seg, "ignore_canopy")
    param_bool(context, p_seg, "ignore_shrubs")
    # odometry parameters
    param_bool(context, p_odom, "confidence_based_insertion")
    param_bool(context, p_odom, "confidence_weighting")
    param_bool(context, p_odom, "correspondence_restriction")
    assert (p := param_int(context, p_odom, "voxel_capacity")) is None or p > 0
    assert (p := param_int(context, p_odom, "max_iterations")) is None or p > 0
    assert (p := param_int(context, p_odom, "max_threads")) is None or p >= 0
    assert (p := param_float(context, p_odom, "threshold_initial")) is None or p > 0.0
    assert (p := param_float(context, p_odom, "min_motion")) is None or p > 0.0
    assert (p := param_float(context, p_odom, "convergence_criterion")) is None or p > 0.0
    assert (p := param_float(context, p_odom, "correspondence_multiplier")) is None or p > 0.0
    
    return [
        Node(
            package=PACKAGE_NAME,
            executable='segmentation_node_offline',
            parameters=[p_seg],
            emulate_tty=True,
            output='screen'
        ),
        Node(
            package=PACKAGE_NAME,
            executable="odometry_node",
            parameters=[p_odom],
            emulate_tty=True,
            output='screen'
        ),
    ]

def generate_launch_description():
    visualize = LaunchConfiguration("visualize")

    return LaunchDescription([
        DeclareLaunchArgument('visualize', default_value="true"),
        # shared parameters
        DeclareLaunchArgument('voxel_size', default_value=""),
        DeclareLaunchArgument('max_distance', default_value=""),
        # segmentation parameters
        DeclareLaunchArgument('path_to_bag'),
        DeclareLaunchArgument('path_to_model', default_value=""),
        DeclareLaunchArgument('incoming_points_topic', default_value=""),
        DeclareLaunchArgument('ignore_canopy', default_value=""),
        DeclareLaunchArgument('ignore_shrubs', default_value=""),
        # odometry parameters
        DeclareLaunchArgument('confidence_based_insertion', default_value=""),
        DeclareLaunchArgument('confidence_weighting', default_value=""),
        DeclareLaunchArgument('correspondence_restriction', default_value=""),
        DeclareLaunchArgument('voxel_capacity', default_value=""),
        DeclareLaunchArgument('max_iterations', default_value=""),
        DeclareLaunchArgument('max_threads', default_value=""),
        DeclareLaunchArgument('threshold_initial', default_value=""),
        DeclareLaunchArgument('min_motion', default_value=""),
        DeclareLaunchArgument('convergence_criterion', default_value=""),
        DeclareLaunchArgument('correspondence_multiplier', default_value=""),
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            condition=IfCondition(visualize),
            arguments=["-d", PathJoinSubstitution([FindPackageShare(PACKAGE_NAME), "rviz", "debug.rviz"])],
            parameters=[{ 'use_sim_time': True }]
        ),
        OpaqueFunction(function=setup)
    ])
