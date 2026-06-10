import sys
import pathlib
import yaml
import enum

# ros
import launch
import launch_ros

# util
sys.path.append(str(pathlib.Path(__file__).parent))
from util.trajectory_analysis import main as trajectory_analysis

PACKAGE_NAME = 'forest_segmentation'

LOGGER = launch.logging.get_logger(pathlib.Path(__file__).name)

PARAM_REQUIRED = [
    'path_to_bag',
    'path_to_ground_truth',
    'path_to_results',
    'lidar_topic',
    'lidar_frame',
    'imu_topic',
    'imu_frame',
]

class Component(enum.Enum):
    SEGMENTATION = 1
    ODOMETRY = 2
    RECORDER = 3

PARAM_DESCRIPTOR = {
    Component.SEGMENTATION: {
        'path_to_bag': 'path_to_bag',
        'path_to_model': None, 
        'incoming_points_topic': 'lidar_topic', 
        'voxel_size': None, 
        'max_distance': None, 
        'ignore_canopy': None,  
        'ignore_shrubs': None,
    },
    Component.ODOMETRY: {
        'voxel_size': None, 
        'max_distance': None,
        'confidence_based_insertion': None, 
        'confidence_weighting': None, 
        'correspondence_restriction': None, 
        'voxel_capacity': None, 
        'max_iterations': None, 
        'max_threads': None, 
        'threshold_initial': None, 
        'min_motion': None, 
        'convergence_criterion': None, 
        'correspondence_multiplier': None, 
    },
    Component.RECORDER: {
        'path_to_record': 'path_to_results'
    }
}

class Config:
    def __init__(self, path_to_config: str):
        path_to_config = pathlib.Path(path_to_config)
        if not path_to_config.exists():
            LOGGER.error(f"path_to_config did not exist: {path_to_config}")
            exit(1)

        with open(path_to_config, "r") as f:
            config_raw = yaml.safe_load(f)

        self.param_root = {}

        missing_required = False
        for key in PARAM_REQUIRED:
            if config_raw.get(key) is None:
                LOGGER.error(f"missing required parameter: {key}")
                missing_required = True
            else:
                self.param_root[key] = config_raw[key]

        if missing_required:
            exit(1)

        def load(param):
            if param.get('config') is not None and param.get('config_file') is not None:
                LOGGER.error("config and config_file are mutually exclusive")
                exit(1)
            if param.get('config') is not None:
                return param['config']
            elif param.get('config_file') is not None:
                path_to_config_inner = pathlib.Path(param['config_file'])
                if not path_to_config_inner.exists():
                    LOGGER.error(f"config_file did not exist: {path_to_config_inner}")
                with open(path_to_config, "r") as f:
                    return yaml.safe_load(f)

        self.param_raw = load(config_raw)

        def load_param_for_component(component):
            param = {}
            for key, val in PARAM_DESCRIPTOR[component].items():
                if val is None:
                    if self.param_raw.get(key) is not None:
                        param[key] = self.param_raw[key]
                elif isinstance(val, str):
                    param[key] = self.param_root[val]
            return param

        self.param = {c: load_param_for_component(c) for c in Component}

    def __getitem__(self, key: Component | str):
        if isinstance(key, Component):
            return self.param[key]
        elif isinstance(key, str):
            return self.param_root[key]

def setup(context, *args, **kwargs):
    path_to_config = launch.substitutions.LaunchConfiguration('path_to_config').perform(context)

    config = Config(path_to_config)

    node_segmentation = launch_ros.actions.Node(
        package = PACKAGE_NAME,
        executable = 'segmentation_node_offline',
        emulate_tty = True,
        output = 'screen',
        parameters = [
            {
                'use_sim_time': True
            },
            config[Component.SEGMENTATION]
        ]
    )

    node_odometry = launch_ros.actions.Node(
        package = PACKAGE_NAME,
        executable = "odometry_node",
        emulate_tty = True,
        output = 'screen',
        parameters = [
            {
                'use_sim_time': True,
                'debug': True
            },
            config[Component.ODOMETRY]
        ]
    )

    node_recorder = launch_ros.actions.Node(
        package = PACKAGE_NAME,
        executable = "recorder_node",
        emulate_tty = True,
        output = 'screen',
        parameters = [
            {
                'use_sim_time': True,
            },
            config[Component.RECORDER]
        ]
    )

    return [
        node_segmentation,
        node_odometry,
        node_recorder,
        launch.actions.RegisterEventHandler(
            event_handler = launch.event_handlers.OnProcessExit(
                target_action = node_recorder,
                on_exit = [
                    launch.actions.EmitEvent(event = launch.events.Shutdown()),
                ]
            )
        ),
        launch.actions.RegisterEventHandler(
            event_handler = launch.event_handlers.OnShutdown(
                on_shutdown = [
                    launch.actions.OpaqueFunction(
                        function = lambda _: trajectory_analysis(
                            path_gt = pathlib.Path(config['path_to_ground_truth']),
                            path_out = pathlib.Path(config['path_to_results'])
                        )
                    )
                ]
            )
        )
    ]

def generate_launch_description():
    visualize = launch.substitutions.LaunchConfiguration('visualize')

    return launch.LaunchDescription([
        launch.actions.SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_zenoh_cpp'),
        launch.actions.DeclareLaunchArgument('visualize', default_value='true'),
        launch.actions.DeclareLaunchArgument('path_to_config'),
        launch_ros.actions.Node(
            package = 'rmw_zenoh_cpp',
            executable = 'rmw_zenohd',
            name = 'router',
        ),
        launch_ros.actions.Node(
            package = 'rviz2',
            executable = 'rviz2',
            output = 'screen',
            condition = launch.conditions.IfCondition(visualize),
            arguments = ['-d', launch.substitutions.PathJoinSubstitution([
                launch_ros.substitutions.FindPackageShare(PACKAGE_NAME), 'rviz', 'debug.rviz'])],
            parameters = [{ 'use_sim_time': True }]
        ),
        launch.actions.OpaqueFunction(function=setup)
    ])
