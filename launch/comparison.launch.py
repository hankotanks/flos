import functools
import pathlib
import yaml
import enum
import tempfile
import csv
import subprocess
import datetime
import copy
import shutil

# numpy
import numpy as np

# ros
import launch
import launch_ros

PACKAGE_NAME = 'forest_segmentation'

LOGGER = launch.logging.get_logger(pathlib.Path(__file__).name)

PARAM_REQUIRED = {
    'path_to_bag': True,
    'path_to_metadata': True,
    'path_to_ground_truth': True,
    'path_to_results': True,
    'lidar_topic': False,
    'lidar_frame': False,
    'imu_topic': False,
    'imu_frame': False,
}

PARAM_OPTIONAL = [
    'ground_truth_format',
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
        'imu_topic': 'imu_topic',
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

def construct_actions_kiss_icp(config):
    cmd = [
        "kiss_icp_pipeline",
        config['path_to_metadata'],
        "--topic", config['lidar_topic'],
    ]
    
    if len(config[Package.KISS_ICP]) > 0:
        path_to_tmp = tempfile.NamedTemporaryFile(mode = "w", suffix = ".yaml", delete = False)
        yaml.safe_dump(config[Package.KISS_ICP], path_to_tmp)
        path_to_tmp.close()

        cmd.extend(["--config", path_to_tmp.name])
        
    action_kiss_icp = launch.actions.ExecuteProcess(
        cwd = str(config['path_to_results']),
        cmd = cmd,
        output = "screen"
    )

    def postprocess() -> pathlib.Path | None:
        path_to_results = pathlib.Path(config['path_to_results'])
        base_to_results_kiss = path_to_results / 'results'

        path_latest = None
        time_latest = None
        for entry in base_to_results_kiss.iterdir():
            if not entry.is_dir(): continue
            try:
                dt = datetime.strptime(entry.name, "%Y-%m-%d_%H-%M-%S")
            except ValueError:
                continue
            if time_latest is None or dt > time_latest:
                time_latest = dt
                path_latest = entry

        if path_latest is None:
            LOGGER.warning("skipping KISS-ICP trajectory")
            return
        
        path_to_tum_src = path_latest / 'rosbags_poses_tum.txt'
        path_to_tum_dst = path_to_results / 'kiss_icp.tum'
        path_to_tum_dst.write_text(path_to_tum_src.read_text())

    actions = [
        launch.actions.RegisterEventHandler(
            event_handler = launch.event_handlers.OnProcessExit(
                target_action = action_kiss_icp,
                on_exit = [
                    launch.actions.OpaqueFunction(function = lambda _: (postprocess(), [])[1])
                ]
            )
        )
    ]

    return {
        'wait_for': action_kiss_icp,
        'actions': actions
    }

class Package(enum.Enum):
    RKO_LIO = (
        "rko_lio", 
        lambda config: {
            'wait_for': None,
            'actions': [
                launch_ros.actions.Node(
                    package = "rko_lio",
                    executable = "online_node",
                    output = "screen",
                    emulate_tty = True,
                    parameters = [
                        {
                            'use_sim_time': True,
                            'base_frame': config['lidar_frame'],
                            'lidar_topic': "/forest_segmentation/raw/lidar",
                            'lidar_frame': config['lidar_frame'],
                            'imu_topic': "/forest_segmentation/raw/imu",
                            'imu_frame': config['imu_frame']
                        },
                        config[Package.RKO_LIO]
                    ],
                )
            ]
        })
    KISS_ICP = (
        "kiss_icp", 
        lambda config: construct_actions_kiss_icp(config))
    
def convert_trajectory_digiforest(path_raw: pathlib.Path) -> pathlib.Path:
    path_tum = (pathlib.Path(tempfile.gettempdir()) / path_raw.name).with_suffix(".tum")

    rows = []
    with open(path_raw, "r") as f:
        reader = csv.DictReader(f, skipinitialspace = True)
        for row in reader:
            timestamp = (float(row["sec"]) + float(row["nsec"]) * 1e-9)
            rows.append([
                timestamp,
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
                float(row["qx"]),
                float(row["qy"]),
                float(row["qz"]),
                float(row["qw"]),
            ])

    np.savetxt(path_tum, np.asarray(rows), fmt = "%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f")
    return path_tum

GROUND_TRUTH_FORMAT = {
    "digiforest": convert_trajectory_digiforest,
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
        for key, is_path in PARAM_REQUIRED.items():
            if config_raw.get(key) is None:
                LOGGER.error(f"missing required parameter: {key}")
                missing_required = True
            else:
                if is_path:
                    path = pathlib.Path(config_raw[key])
                    if path.is_absolute():
                        self.param_root[key] = str(path.resolve())
                    else:
                        self.param_root[key] = str((path_to_config.parent / path).resolve())
                else:
                    self.param_root[key] = config_raw[key]

        if missing_required:
            exit(1)

        for key in PARAM_OPTIONAL:
            if config_raw.get(key) is None:
                continue

            self.param_root[key] = config_raw[key]

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
            return {}

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

        self.param_packages = {p: None for p in Package}
        for package in Package:
            if config_raw.get(package.value[0]) is None: continue
            if config_raw[package.value[0]].get('active') is None: continue
            if config_raw[package.value[0]]['active']:
                self.param_packages[package] = load(config_raw[package.value[0]])

    def __getitem__(self, key: Component | Package | str):
        if isinstance(key, Component):
            return self.param[key]
        elif isinstance(key, Package):
            return self.param_packages[key]
        elif isinstance(key, str):
            if self.param_root.get(key) is None:
                return None
            return self.param_root[key]
        
    def __setitem__(self, key: Component | Package | str, value):
        if isinstance(key, Component):
            self.param[key] = value
        elif isinstance(key, Package):
            self.param_packages[key] = value
        elif isinstance(key, str):
            self.param_root[key] = value

    def save(self, path_to_config: pathlib.Path):
        with open(path_to_config, 'w') as f:
            yaml.safe_dump(self.param_root, f)

            param_config = {}
            for component in Component:
                for key, key_map in PARAM_DESCRIPTOR[component].items():
                    if param_config.get(key) is not None: 
                        continue
                    if key_map is not None:
                        continue
                    if (p := self.param[component].get(key)) is not None:
                        param_config[key] = p

            if len(param_config) > 0:
                yaml.safe_dump({ 'config': param_config }, f)
            
            for package in Package:
                if self.param_packages[package] is not None:
                    param_package = { 
                        'active': True 
                    }

                    if len(self.param_packages[package]) > 0:
                        param_package['config'] = self.param_packages[package]

                    yaml.safe_dump({ f'{package.value[0]}': param_package }, f)
        
def analysis(context, config):
    path_to_ground_truth = pathlib.Path(config['path_to_ground_truth'])
    if not path_to_ground_truth.exists():
        LOGGER.error(f"path_to_ground_truth does not exist: {path_to_ground_truth}")
        exit(1)

    if (fmt := config['ground_truth_format']) is not None:
        if GROUND_TRUTH_FORMAT.get(fmt) is None:
            LOGGER.error(f"unrecognized ground truth format [must be one of {", ".join(GROUND_TRUTH_FORMAT.keys())}]")
            exit(1)
        
        path_to_ground_truth = GROUND_TRUTH_FORMAT[fmt](path_to_ground_truth)
        
    path_to_results = pathlib.Path(config["path_to_results"])
    if not path_to_results.exists():
        LOGGER.error(f"path_to_results did not exist: {path_to_results}")
        exit(1)

    path_to_evo = path_to_results / "evo"
    if path_to_evo.exists():
        shutil.rmtree(path_to_evo)
    path_to_evo.mkdir(parents = True, exist_ok = True)

    path_to_tum = [p for p in path_to_results.glob("*.tum") if p.resolve() != path_to_ground_truth.resolve()]

    subprocess.run([
        "evo_traj",
        "tum",
        *map(str, path_to_tum),
        "--ref", str(path_to_ground_truth),
        "--align",
        "--plot_mode", "xy",
        "--save_plot", str(path_to_evo / "traj_xy.pdf"),
    ], check = True)

    for path_to_tum_i in path_to_tum:
        subprocess.run([
            "evo_ape",
            "tum",
            str(path_to_ground_truth),
            str(path_to_tum_i),
            "--align",
            "--correct_scale",
            "--save_results",
            str(path_to_evo / f"{path_to_tum_i.stem}_ape.zip"),
        ], check = True)

    for path_to_tum_i in path_to_tum:
        subprocess.run([
            "evo_rpe",
            "tum",
            str(path_to_ground_truth),
            str(path_to_tum_i),
            "--align",
            "--correct_scale",
            "--save_results",
            str(path_to_evo / f"{path_to_tum_i.stem}_rpe.zip"),
        ], check = True)

    subprocess.run([
        "evo_res",
        *map(str, path_to_evo.glob("*_ape.zip")),
        "--use_filenames",
        "--save_plot", str(path_to_evo / "ape_summary.pdf"),
        "--save_table", str(path_to_evo / "ape_table.csv")
    ], check = True)

    subprocess.run([
        "evo_res",
        *map(str, path_to_evo.glob("*_rpe.zip")),
        "--use_filenames",
        "--save_plot", str(path_to_evo / "rpe_summary.pdf"),
        "--save_table", str(path_to_evo / "rpe_table.csv")
    ], check = True)

def cleanup(config):
    config = copy.deepcopy(config)

    path_to_results = pathlib.Path(config['path_to_results']) 

    path_to_ground_truth_src = pathlib.Path(config['path_to_ground_truth'])
    config['path_to_ground_truth'] = './ground_truth' + path_to_ground_truth_src.suffix
    path_to_ground_truth_dst = path_to_results / ('ground_truth' + path_to_ground_truth_src.suffix)
    path_to_ground_truth_dst.write_text(path_to_ground_truth_src.read_text())

    config['path_to_results'] = '.'

    config.save(path_to_results / f'{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.yaml')

def setup(context, *args, **kwargs):
    visualize = launch.substitutions.LaunchConfiguration('visualize')

    path_to_config = launch.substitutions.LaunchConfiguration('path_to_config').perform(context)
    config = Config(path_to_config)

    action_cleanup = launch.actions.OpaqueFunction(function = lambda _: (cleanup(config), [])[1])

    action_analysis = launch.actions.OpaqueFunction(function = functools.partial(analysis, config = config))

    on_exit = [action_analysis, action_cleanup]
    if launch.substitutions.LaunchConfiguration('reanalyze').perform(context) == 'true':
        return on_exit

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

    description = [
        node_segmentation,
        node_odometry,
        node_recorder,
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
    ]

    # nodes which should be shutdown after the recorder_node finishes
    nodes = [
        node_odometry
    ]

    chained_actions = []
    for package in Package:
        if config[package] is None:
            continue
        
        result = package.value[1](config)
        if (action := result['wait_for']) is not None:
            chained_actions.append((action, result['actions']))
        else:
            for action in result['actions']:
                if isinstance(action, launch_ros.actions.Node):
                    nodes.append(action)
                description.append(action)

    for waiter, actions in reversed(chained_actions):
        on_exit_new = [
            waiter,
        ]

        if len(on_exit) > 0 or actions > 0:
            on_exit_new.append(launch.actions.RegisterEventHandler(
                event_handler = launch.event_handlers.OnProcessExit(
                    target_action = waiter,
                    on_exit = actions + on_exit
                )
            ))

        on_exit = on_exit_new

    if len(on_exit) > 0:
        description.append(launch.actions.RegisterEventHandler(
            event_handler = launch.event_handlers.OnProcessExit(
                target_action = node_recorder,
                on_exit = on_exit
            )
        ))

    
    description.append(launch.actions.RegisterEventHandler(
        event_handler = launch.event_handlers.OnProcessExit(
            target_action = node_recorder,
            on_exit = [
                launch.actions.EmitEvent(event = launch.events.process.ShutdownProcess(process_matcher = lambda process: process in nodes))
            ]
        )
    ))

    return description

def generate_launch_description():
    return launch.LaunchDescription([
        launch.actions.SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_zenoh_cpp'),
        launch.actions.DeclareLaunchArgument('visualize', default_value = 'true'),
        launch.actions.DeclareLaunchArgument('reanalyze', default_value = 'false'),
        launch.actions.DeclareLaunchArgument('path_to_config'),
        launch.actions.OpaqueFunction(function = setup)
    ])
