import enum
import pathlib
import yaml
import typing

import launch
import launch_ros

class Config:
    class Component(enum.Enum):
        OFFLINE = {
            'path_to_data': ((pathlib.Path, True), True),
            'path_to_result': ((pathlib.Path, False), False),
            'frames': (int, False),
            'visualize': (bool, False),
            'visualize_map': (bool, False),
            'incoming_points_topic': (str, False), # NOTE: specific to BagDataloader
        }
        SEGMENTATION = {
            'max_distance': (float, False), 
            'voxel_size': (float, False), 
            'voxel_size_inference': (float, False), 
            'double_downsample': (bool, False),
            'semantics_ignore': (typing.List[int], False),
            'semantics_remap': (typing.Dict[int, int], False),
            'path_to_model': ((pathlib.Path, True), False), 
        }
        ODOMETRY = {
            'max_distance': (float, False), 
            'max_distance_ground': (float, False), 
            'voxel_size': (float, False), 
            'voxel_capacity': (int, False), 
            'threshold_initial': (float, False), 
            'threshold_motion': (float, False), 
            'deskew': (bool, False),
            'double_downsample': (bool, False),
            'constant_scan_rate': (bool, False),
            'max_iterations': (int, False), 
            'max_threads': (int, False), 
            'convergence_criterion': (float, False),
            'correspondence_weighting': (bool, False), 
            'correspondence_target': (int, False),
            'projection_enable': (bool, False),
            'projection_iou': (float, False),
            'projection_ground': (typing.List[int], False),
            'projection_target': (typing.List[int], False)
        }

    def __init__(self, context):
        path_to_config = launch.substitutions.LaunchConfiguration('path_to_config').perform(context)

        self.param = {component: {} for component in self.__class__.Component}

        def parse(ty, val, key, cwd = pathlib.Path.cwd()):
            if val is None: return None
            strict = False
            if isinstance(ty, tuple): ty, strict = ty
            origin = typing.get_origin(ty)
            args = typing.get_args(ty)
            if origin is list:
                if not isinstance(val, (list, tuple)):
                    if isinstance(val, str): return parse(ty, yaml.safe_load(val), key, cwd)
                    raise RuntimeError(f'{key} must be a list: {val}')
                subtype = args[0]
                return [parse(subtype, x, key, cwd) for x in val]
            if origin is dict:
                if not isinstance(val, dict):
                    if isinstance(val, str):
                        return parse(ty, yaml.safe_load(val), key, cwd)
                    raise RuntimeError(f'{key} must be a dictionary: {val}')
                key_ty, value_ty = args
                keys = []
                vals = []
                for k, v in val.items():
                    keys.append(parse(key_ty, k, key, cwd))
                    vals.append(parse(value_ty, v, key, cwd))
                return (keys, vals)
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

            for component in self.__class__.Component:
                for key in component.value.keys():
                    ty, _ = component.value[key]
                    if (p := parse(ty, param_from_config.get(key), key, path_to_config.parent)) is not None:
                        if isinstance(p, tuple):
                            k, v = p
                            self.param[component][key + "_key"] = k
                            self.param[component][key + "_val"] = v
                        else:
                            self.param[component][key] = p

        for component in self.__class__.Component:
            for key in component.value.keys():
                ty, req = component.value[key]
                val = launch.substitutions.LaunchConfiguration(key).perform(context)
                val = val if len(val) > 0 else None

                if (p := parse(ty, val, key)) is not None:
                    if isinstance(p, tuple):
                        k, v = p
                        self.param[component][key + "_key"] = k
                        self.param[component][key + "_val"] = v
                    else:
                        self.param[component][key] = p
                elif req and self.param[component].get(key) is None:
                    raise RuntimeError(f'missing required parameter: {key} [{ty}]')
                        
    @classmethod
    def declare_launch_arguments(cls):
        processed = set()
        arguments = [
            launch.actions.DeclareLaunchArgument('path_to_config', default_value = '')
        ]
        for component in cls.Component:
            for key in component.value.keys():
                if key in processed:
                    continue
                arguments.append(launch.actions.DeclareLaunchArgument(key, default_value = ''))
                processed.add(key)
        
        return arguments
    
    def __getitem__(self, component: Component):
        if not isinstance(component, self.__class__.Component):
            raise IndexError(f'invalid index')
        return self.param[component]
