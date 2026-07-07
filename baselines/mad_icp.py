import sys
import pathlib
import subprocess
import numpy as np
import yaml
import rosbags
import rosbags.highlevel
import scipy
import scipy.spatial
import scipy.spatial.transform

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from util.loader import Loader

# example
# python3 baselines/mad_icp.py digiforests ../digi_ros2/ ../digi_ros2/sma/results/ --path_to_config ../mad_icp_50m.cfg --name mad_icp_50m

def read_topic(config_path: pathlib.Path) -> str:
    with open(config_path, "r") as f: config = yaml.safe_load(f)
    if config.get('rosbag_topic') is None: raise ValueError(f'failed to read rosbag_topic')
    return config["rosbag_topic"]

def read_stamps(bag_folder: pathlib.Path, topic: str) -> list[float]:
    stamps = []
    with rosbags.highlevel.AnyReader([bag_folder]) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections: raise ValueError(f"topic '{topic}' not found.")
        for _, ns, _ in reader.messages(connections = connections): stamps.append(ns * 1e-9)
    return stamps

def convert_trajectory(kitti_path: pathlib.Path, tum_path: pathlib.Path, stamps: list) -> None:
    poses = np.loadtxt(kitti_path)
    if poses.ndim == 1: poses = poses[None, :]
    if poses.shape[1] != 12: raise ValueError(f"invalid entry [expected 12 values, found {poses.shape[1]}]")
    n = len(poses)
    if len(stamps) != n: raise ValueError("number of timestamps doesn't match pose count")
    with open(tum_path, "w") as f:
        for t, pose in zip(stamps, poses):
            T = pose.reshape(3, 4)
            R = T[:, :3]
            x, y, z = T[:, 3]
            qx, qy, qz, qw = scipy.spatial.transform.Rotation.from_matrix(R).as_quat()
            f.write(f"{t:.9f} {x:.9f} {y:.9f} {z:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")

def main():
    parser = Loader.parser(require_path_to_config = True)
    args = parser.parse_args()
    for _, config, bag, metadata, result in Loader(args):
        data_path = bag.parent
        if data_path != metadata.parent: raise RuntimeError("bag and metadata must share the same parent")
        topic = read_topic(config)
        stamps = read_stamps(bag.parent, topic)
        if len(stamps) == 0: raise RuntimeError("unable to read timstamps")
        # construct command (with inline configuration string if provided)
        cmd = f'mad_icp --data-path {data_path} --estimate-path {result} --dataset-config {config} --noviz'
        # run mad-icp
        subprocess.run(cmd, cwd = result, shell = True)
        # copy trajectory to correct location
        path_to_traj_src = result / 'estimate.txt'
        path_to_traj_dst = result / 'mad_icp_tum.txt'
        convert_trajectory(path_to_traj_src, path_to_traj_dst, stamps)

if __name__ == '__main__':
    main()