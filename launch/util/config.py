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
            'frames': (int, False),
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

        def parse(component, key, val, cwd = pathlib.Path.cwd()):
            if val is None:
                return None
            
            ty, _ = component.value[key]
            if ty is str:
                return val
            elif isinstance(ty, tuple):
                ty, strict = ty
                if ty is not pathlib.Path:
                    raise AssertionError('invalid component parameters')
                val_abs = pathlib.Path(val)
                try:
                    return str(val_abs.resolve(strict = strict))
                except:
                    try:
                        val_abs = (cwd / val_abs)
                        return str(val_abs.resolve(strict = strict))
                    except:
                        raise RuntimeError(f'{key} could not be resolved: {val}')
            elif ty is bool:
                if isinstance(val, bool):
                    return val
                elif not isinstance(val, str):
                    raise AssertionError('unreachable')
                if val not in ['true', 'false']:
                    raise RuntimeError(f'{key} must be a boolean [true, false]: {val}')
                return True if val == 'true' else False
            elif ty is int:
                if isinstance(val, int):
                    return val
                elif not isinstance(val, str):
                    raise AssertionError('unreachable')
                try:
                    return int(val)
                except:
                    raise RuntimeError(f'{key} must be an integer: {val}')
            elif ty is float:
                if isinstance(val, float):
                    return val
                elif not isinstance(val, str):
                    raise AssertionError('unreachable')
                try:
                    return float(val)
                except:
                    raise RuntimeError(f'{key} must be a number: {val}')
            else:
                raise AssertionError('invalid component parameters')
    
        if path_to_config is not None and len(path_to_config) > 0:
            param_from_config = {}
            try:
                path_to_config = pathlib.Path(path_to_config).resolve(strict = True)
                with open(path_to_config, 'r') as f:
                    param_from_config = yaml.safe_load(f)
            except:
                raise RuntimeError(f'path_to_config does not exist: {path_to_config}')

            for component in Config.Component:
                for key in component.value.keys():
                    if (p := parse(component, key, param_from_config.get(key), path_to_config.parent)) is not None:
                        self.param[component][key] = p

        for component in Config.Component:
            for key in component.value.keys():
                val = launch.substitutions.LaunchConfiguration(key).perform(context)
                val = val if len(val) > 0 else None
                if (p := parse(component, key, val)) is not None:
                    self.param[component][key] = p
                else:
                    ty, req = component.value[key]
                    if req and self.param[component].get(key) is None:
                        raise RuntimeError(f'missing required parameter: {key} [{ty.__name__}]')

                
    def declare_launch_arguments():
        processed = set()
        arguments = [
            launch.actions.DeclareLaunchArgument('path_to_config', default_value = '')
        ]
        for component in Config.Component:
            for key in component.value.keys():
                if key in processed:
                    continue
                arguments.append(launch.actions.DeclareLaunchArgument(key, default_value = ''))
                processed.add(key)
        
        return arguments
    
    def __getitem__(self, component: Component):
        if not isinstance(component, Config.Component):
            raise IndexError(f'invalid index')
        return self.param[component]