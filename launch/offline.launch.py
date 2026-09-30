import pathlib
import sys

# ros
import launch
import launch_ros

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from util.config import Config

PACKAGE_NAME = 'forest_segmentation'

LOGGER = launch.logging.get_logger(pathlib.Path(__file__).name)

def setup(context, *args, **kwargs):
    try:
        config = Config(context)
    except RuntimeError as e:
        LOGGER.error(e.args[0])
        exit(1)

    node_offline = launch_ros.actions.Node(
        package = PACKAGE_NAME,
        executable = 'offline_node',
        emulate_tty = True,
        output = 'screen',
        parameters = [
            {
                'use_sim_time': True,
            },
            config[Config.Component.OFFLINE] | config[Config.Component.SEGMENTATION] | config[Config.Component.ODOMETRY]
        ]
    )

    description = [
        launch_ros.actions.Node(
            package = 'rmw_zenoh_cpp',
            executable = 'rmw_zenohd',
            name = 'router',
        ),
        node_offline,
        launch.actions.RegisterEventHandler(
            event_handler = launch.event_handlers.OnProcessExit(
                target_action = node_offline,
                on_exit = [
                    launch.actions.EmitEvent(event = launch.events.Shutdown()),
                ]
            )
        )
    ]

    if 'visualize' in config[Config.Component.OFFLINE] and config[Config.Component.OFFLINE]['visualize']:
        description.append(launch_ros.actions.Node(
            package = 'rviz2',
            executable = 'rviz2',
            output = 'screen',
            arguments = ['-d', launch.substitutions.PathJoinSubstitution([
                launch_ros.substitutions.FindPackageShare(PACKAGE_NAME), 'launch', 'rviz', 'default.rviz'])],
            parameters = [{ 'use_sim_time': True }]
        ))

    return description

def generate_launch_description():
    return launch.LaunchDescription([
        launch.actions.SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_zenoh_cpp'),
        *Config.declare_launch_arguments(),
        launch.actions.OpaqueFunction(function = setup)
    ])
