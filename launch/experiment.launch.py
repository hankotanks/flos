from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetEnvironmentVariable
from launch_ros.substitutions import FindPackageShare
from launch.actions import OpaqueFunction

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

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

def create_kiss_icp_config(p_odom, config_file):
    with open(config_file, "w") as f:
        # data
        f.write('data\n')
        f.write('  deskew: True\n')
        if (p := p_odom.get('max_distance')) is not None:
            f.write(f'  max_range: {p}\n')
            f.write(f'  min_range: 0.0\n')
        # mapping
        f.write('mapping:\n')
        if (p := p_odom.get('voxel_size')) is not None:
            f.write(f'  voxel_size: {p}\n')
        if (p := p_odom.get('voxel_capacity')) is not None:
            f.write(f'  max_points_per_voxel: {p}\n')
        # adaptive_threshold
        f.write('adaptive_threshold:\n')
        if (p := p_odom.get('threshold_initial')) is not None:
            f.write(f'  initial_threshold: {p}\n')
        if (p := p_odom.get('min_motion')) is not None:
            f.write(f'  min_motion_th: {p}\n')
        # registration
        f.write('registration:\n')
        if (p := p_odom.get('max_iterations')) is not None:
            f.write(f'  max_num_iterations: {p}\n')
        if (p := p_odom.get('convergence_criterion')) is not None:
            f.write(f'  convergence_criterion: {p}\n')
        if (p := p_odom.get('max_threads')) is not None:
            f.write(f'  max_num_threads: {p}\n')

def get_base_frame(bag_path, topic_name):
    reader = SequentialReader()
    storage_options = StorageOptions(uri = bag_path, storage_id = "mcap" if bag_path.endswith(".mcap") else "sqlite3")
    converter_options = ConverterOptions(input_serialization_format = "cdr", output_serialization_format = "cdr")
    reader.open(storage_options, converter_options)
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == topic_name:
            msg = deserialize_message(data, PointCloud2)
            return msg.header.frame_id
    return None

def setup(context, *args, **kwargs):
    p_seg = {
        'use_sim_time': True
    }

    p_odom = {
        'use_sim_time': True,
        'debug': True
    }

    p_kiss = {
        'use_sim_time': True,
        'visualize': False,
        'config_file': "/tmp/kiss_icp.yaml"
    }

    p_rko_lio = {
        'use_sim_time': True,
        'lidar_topic': "/forest_segmentation/raw/lidar",
        'imu_topic': "/forest_segmentation/raw/imu"
    }

    p_recorder = {
        'use_sim_time': True,
    }

    rko_lio_disable = True
    
    # shared parameters 
    assert (p := param_float(context, [p_seg, p_odom], "voxel_size")) is None or p > 0.0
    assert (p := param_float(context, [p_seg, p_odom], "max_distance")) is None or p > 0.0
    if param_str(context, p_seg, "imu_topic") is not None:
        rko_lio_disable = False

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

    # kiss_icp configuration
    create_kiss_icp_config(p_odom, p_kiss['config_file'])

    # rko_lio parameters
    if not rko_lio_disable:
        # base_frame = get_base_frame(p_seg["path_to_bag"], p_seg["incoming_points_topic"])
        # assert base_frame is not None
        p_rko_lio['base_frame'] = "hesai_lidar_T_BL" # TODO: get this value programmatically
        p_rko_lio['imu_frame'] = "alphasense_imu_T_BI" # TODO: get this value programmatically
        p_rko_lio['lidar_frame'] = "hesai_lidar_T_BL" # TODO: get this value programmatically

    # recorder parameters
    param_str(context, p_recorder, "path_to_record")

    nodes = [
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
        Node(
            package="kiss_icp",
            executable="kiss_icp_node",
            name="kiss_icp_node",
            parameters=[p_kiss],
            emulate_tty=True,
            output="screen",
            remappings=[
                ("pointcloud_topic", "/forest_segmentation/raw/lidar"),
            ]
        ),
        Node(
            package=PACKAGE_NAME,
            executable="recorder_node",
            parameters=[p_recorder],
            emulate_tty=True,
            output='screen'
        )
    ]

    if not rko_lio_disable:
        nodes.extend([
            Node(
                package="rko_lio",
                executable="online_node",
                parameters=[p_rko_lio],
                output="screen",
                emulate_tty=True,
            )
        ])
    
    return nodes

def generate_launch_description():
    visualize = LaunchConfiguration("visualize")

    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_zenoh_cpp'),
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
        # recorder parameters
        DeclareLaunchArgument('path_to_record', default_value=""),
        # rko_lio parameters
        DeclareLaunchArgument('imu_topic', default_value=""),
        Node(
            package='rmw_zenoh_cpp',
            executable='rmw_zenohd',
            name='router',
        ),
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
