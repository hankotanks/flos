import enum
import pathlib
import yaml

import launch
import launch_ros

class Config:
    class Component(enum.Enum):
        OFFLINE = {
            'incoming_points_topic': (str, True),
            'incoming_imu_topic': (str, False),
            'path_to_bag': ((pathlib.Path, True), True),
            'path_to_result': ((pathlib.Path, False), True),
        }
        SEGMENTATION = {
            'path_to_model': ((pathlib.Path, True), False), 
            'voxel_size': (float, False), 
            'max_distance': (float, False), 
            'ignore_canopy': (bool, False),  
            'ignore_shrubs': (bool, False),
        }
        ODOMETRY = {
            'voxel_size': (float, False), 
            'max_distance': (float, False),
            'confidence_based_insertion': (bool, False), 
            'confidence_weighting': (bool, False), 
            'correspondence_restriction': (bool, False), 
            'voxel_capacity': (int, False), 
            'max_iterations': (int, False), 
            'max_threads': (int, False), 
            'threshold_initial': (float, False), 
            'min_motion': (float, False), 
            'convergence_criterion': (float, False), 
            'correspondence_multiplier': (float, False), 
        }

    def __init__(self, context):
        path_to_config = launch.substitutions.LaunchConfiguration('path_to_config').perform(context)

        self.param = {component: {} for component in Config.Component}
        
        if path_to_config is not None and len(path_to_config) > 0:
            param_from_config = {}
            try:
                path_to_config = pathlib.Path(path_to_config).resolve(strict = True)
                with open(path_to_config, 'w') as f:
                    param_from_config = yaml.safe_load(f)
            except:
                raise RuntimeError(f'path_to_config does not exist: {path_to_config}')
            
            def parse(component, key, val):
                ty, req = component.value[key]
                if val is None:
                    if req:
                        raise RuntimeError(f'missing required parameter: {key} [{ty.__name__}]')
                    return None

                if isinstance(ty, str):
                    return val
                elif isinstance(ty, tuple):
                    ty, strict = ty
                    if not isinstance(ty, pathlib.Path):
                        raise AssertionError(f'invalid configuration')
                    try:
                        return (path_to_config.parent / pathlib.Path(val)).resolve(strict = strict)
                    except:
                        raise RuntimeError(f'{key} does not exist: {val}')
                elif isinstance(ty, bool):
                    if val not in ['true', 'false']:
                        raise RuntimeError(f'{key} must be a boolean [true, false]')
                    self.param[component][key] = True if val == 'true' else False
                elif isinstance(ty, int):
                    try:
                        return int(val)
                    except:
                        raise RuntimeError(f'{key} must be an integer: {val}')
                elif isinstance(ty, float):
                    try:
                        return float(val)
                    except:
                        raise RuntimeError(f'{key} must be a number: {val}')

            for component in Config.Component:
                for key in component.value.keys():
                    if (p := parse(component, key, param_from_config.get(key))) is not None:
                        self.param[component][key] = p

        for component in Config.Component:
            for key in component.value.keys():
                if (p := parse(component, key, launch.substitutions.LaunchConfiguration(key).perform(context))) is not None:
                    self.param[component][key] = p
                
    def declare_launch_arguments():
        processed = {}
        arguments = [
            launch.actions.DeclareLaunchArgument('path_to_config', default_value = '')
        ]
        for component in Config.Component:
            for key, (_, req) in component.value.items():
                if key in processed:
                    continue
                if req:
                    arguments.append(launch.actions.DeclareLaunchArgument(key))
                else:
                    arguments.append(launch.actions.DeclareLaunchArgument(key, default_value = ''))
                processed.add(key)
        
        return arguments
    
    def __getitem__(self, component: Component):
        if not isinstance(component, Config.Component):
            raise IndexError(f'invalid index')
        return self.param[component]