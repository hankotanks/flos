import json
from pathlib import Path

import yaml

def load_yaml(fp: Path):
    """load yaml file at fp"""
    if not fp.exists() or not fp.is_file():
        raise Exception(f"invalid fp {fp}, check")
    with fp.open("r") as yaml_file:
        res = yaml.safe_load(yaml_file)
    return res

def load_json(fp: Path):
    """load json file at fp"""
    if not fp.exists() or not fp.is_file():
        raise Exception(f"invalid fp {fp}, check")
    with fp.open("r") as json_file:
        res = json.load(json_file)
    return res

def load_yaml_or_json(fp: Path):
    """loads file at fp based on extension"""
    if not fp.exists() or not fp.is_file():
        raise Exception(f"invalid fp {fp}, check")
    if fp.suffix == ".yaml" or fp.suffix == ".yml":
        return load_yaml(fp)
    elif fp.suffix == ".json":
        return load_json(fp)
    else:
        raise Exception(f"check the extension for {fp}")