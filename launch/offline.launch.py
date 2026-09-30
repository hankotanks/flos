# system
import pathlib
import sys

# ros
import launch
import launch_ros

# package
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from util.config import Config

PACKAGE_NAME = 'flos'

LOGGER = launch.logging.get_logger(pathlib.Path(__file__).name)

def setup(context, *args, **kwargs):
    try:
        config = Config(context)
    except RuntimeError as e:
        LOGGER.error(e.args[0])
        exit(1)

    gdb = launch.substitutions.LaunchConfiguration('gdb').perform(context)
    gdb = True if gdb is not None and gdb == 'true' else False

    node_offline = launch_ros.actions.Node(
        package = PACKAGE_NAME,
        executable = 'offline_rerun_node' if config[Config.Component.OFFLINE]['visualize'] else 'offline_node',
        emulate_tty = True,
        output = 'screen',
        prefix = [f'gdb -batch -x {pathlib.Path(__file__).parent / "util" / "debug.gdb"} --args'] if gdb else None,
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

    return description

def generate_launch_description():
    return launch.LaunchDescription([
        launch.actions.SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_zenoh_cpp'),
        launch.actions.DeclareLaunchArgument('gdb', default_value = 'false'),
        *Config.declare_launch_arguments(),
        launch.actions.OpaqueFunction(function = setup)
    ])
