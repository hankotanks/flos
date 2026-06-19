import pathlib

import launch
import ament_index_python

PACKAGE_NAME = 'forest_segmentation'

LOGGER = launch.logging.get_logger(pathlib.Path(__file__).name)

def chain(actions):
    description = []
    if len(actions) == 0:
        return description
    description.append(actions[0])
    if len(actions) == 1:
        return description

    for action_prev, action_curr in zip(actions, actions[1:]):
        description.append(launch.actions.RegisterEventHandler(
            launch.event_handlers.OnProcessExit(
                target_action = action_prev,
                on_exit = [
                    action_curr
                ]
            )
        ))

    return description

def resolve_path_argument(context, name, required = False, exists = False):
    raw = launch.substitutions.LaunchConfiguration(name).perform(context)
    if not required and (raw is None or len(raw) == 0):
        return None
    try:
        return pathlib.Path(raw).resolve(strict = exists)
    except:
        raise RuntimeError(f'failed to resolve {name}: {raw}')
    
def resolve_bag(base_path: pathlib.Path):
    mcap_files = list(base_path.rglob("*.mcap"))
    if not mcap_files: return None
    if len(mcap_files) > 1:
        LOGGER.warning(f'expected a single mcap rosbag, but found multiple: {base_path}')
        return None
    return mcap_files[0].resolve(), base_path.name

def setup(context, *args, **kwargs):
    try:
        path_to_bags = resolve_path_argument(context, 'path_to_bags', required = True, exists = True)
        path_to_config = resolve_path_argument(context, 'path_to_config', required = True, exists = True)
        path_to_result = resolve_path_argument(context, 'path_to_result', required = True, exists = False)
    except RuntimeError as e:
        LOGGER.error(e.args[0])
        exit(1)

    path_to_bags = [resolve_bag(p) for p in path_to_bags.iterdir() if p.is_dir() and p not in path_to_result.parents]
    path_to_bags = list(filter(lambda p: p is not None, path_to_bags))

    path_to_launch = ament_index_python.packages.get_package_share_directory(PACKAGE_NAME)
    path_to_launch = pathlib.Path(path_to_launch).resolve(strict = True)
    path_to_launch = path_to_launch / 'launch' / 'offline.launch.py'

    name = launch.substitutions.LaunchConfiguration('name').perform(context)
    if name is None or len(name) == 0:
        name = path_to_config.stem

    configs = []
    if path_to_config.is_dir():
        for config in list(path_to_config.glob('*.yaml')):
            configs.append((config, f'{name}_{config.stem}'))
    else:
        configs.append((path_to_config, name))

    def create_actions(config, config_name):
        path_to_result_inner = path_to_result / config_name
        path_to_result_inner.mkdir(parents = True, exist_ok = True)
        path_to_result_inner = path_to_result_inner.resolve(strict = True)

        # copy the odometry and sma config files
        if path_to_result_inner != config.parent and config_name != config.stem:
                (path_to_result_inner / (config_name + '.yaml')).write_text(config.read_text())
        (path_to_result_inner / 'config.yaml').write_text('category: odometry')

        return [
            launch.actions.ExecuteProcess(
                cmd = [
                    'ros2', 'launch', str(path_to_launch),
                    'visualize:=false',
                    f'path_to_bag:={str(bag)}',
                    f'path_to_result:={str(path_to_result_inner / "digiforests" / bag_name)}',
                    f'path_to_config:={str(config)}'
                ],
                output = 'screen',
                emulate_tty = True,
            ) for bag, bag_name in path_to_bags
        ]
    
    actions = []
    for config, config_name in configs:
        actions.extend(create_actions(config, config_name))

    description = [
        *chain(actions)
    ]

    return description

def generate_launch_description():
    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument('path_to_bags'),
        launch.actions.DeclareLaunchArgument('path_to_config'),
        launch.actions.DeclareLaunchArgument('path_to_result'),
        launch.actions.DeclareLaunchArgument('name', default_value = ''),
        launch.actions.OpaqueFunction(function = setup)
    ])
