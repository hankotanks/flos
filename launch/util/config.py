import enum
import pathlib
import yaml
import typing

import launch
import launch_ros

class Config:
    class Component(enum.Enum):
        OFFLINE = {
            'incoming_points_topic': (str, True),
            'path_to_bag': ((pathlib.Path, True), True),
            'path_to_result': ((pathlib.Path, False), False),
            'frames': (int, False),
        }
        SEGMENTATION = {
            'path_to_model': ((pathlib.Path, True), False), 
            'voxel_size': (float, False), 
            'max_distance': (float, False), 
            'max_distance_neighborhood': (float, False), 
            'ignore': (typing.List[int], False),
            'nearby': (typing.List[int], False),
        }
        ODOMETRY = {
            'voxel_size': (float, False), 
            'max_distance': (float, False), 
            'min_weight': (float, False),
            'confidence_weighting': (bool, False), 
            'correspondence_restriction': (bool, False), 
            'correspondence_with_voxel_belief': (bool, False),
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

        def parse(ty, val, key, cwd=pathlib.Path.cwd()):
            if val is None: return None
            strict = False
            if isinstance(ty, tuple): ty, strict = ty
            origin = typing.get_origin(ty)
            args = typing.get_args(ty)
            if origin is list:
                if not isinstance(val, (list, tuple)): raise RuntimeError(f'{key} must be a list: {val}')
                subtype = args[0]
                return [parse(subtype, x, key, cwd) for x in val]
            if ty is pathlib.Path:
                val_abs = pathlib.Path(val)
                try:
                    return str(val_abs.resolve(strict=strict))
                except FileNotFoundError:
                    try:
                        return str((cwd / val_abs).resolve(strict=strict))
                    except FileNotFoundError:
                        raise RuntimeError(f'{key} could not be resolved: {val}')
            if ty is str: return str(val)
            if ty is bool:
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    if val == "true":
                        return True
                    if val == "false":
                        return False
                raise RuntimeError(f'{key} must be a boolean [true, false]: {val}')
            if ty is int:
                if isinstance(val, int): return val
                try:
                    return int(val)
                except Exception:
                    raise RuntimeError(f'{key} must be an integer: {val}')
            if ty is float:
                if isinstance(val, float): return val
                try:
                    return float(val)
                except Exception:
                    raise RuntimeError(f'{key} must be a number: {val}')
            raise AssertionError(f'invalid parameter type: {ty}')
        
        if path_to_config is not None and len(path_to_config) > 0:
            param_from_config = {}
            try:
                path_to_config = pathlib.Path(path_to_config).resolve(strict=True)
                with open(path_to_config, 'r') as f:
                    param_from_config = yaml.safe_load(f)
            except Exception:
                raise RuntimeError(f'path_to_config does not exist: {path_to_config}')

            for component in Config.Component:
                for key in component.value.keys():
                    ty, _ = component.value[key]
                    if (p := parse(ty, param_from_config.get(key), key, path_to_config.parent)) is not None:
                        self.param[component][key] = p


        for component in Config.Component:
            for key in component.value.keys():
                ty, req = component.value[key]
                val = launch.substitutions.LaunchConfiguration(key).perform(context)
                val = val if len(val) > 0 else None

                if (p := parse(ty, val, key)) is not None:
                    self.param[component][key] = p
                elif req and self.param[component].get(key) is None:
                    raise RuntimeError(f'missing required parameter: {key} [{ty}]')
                        
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