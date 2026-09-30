import sys
import pathlib
import subprocess

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from util.loader import Loader, resolve

# example
# python3 baselines/genz_icp.py digiforests ../digi_ros2/ ../digi_ros2/sma/results/ --config '{"max_range": 50.0}' --name genz_icp_50m

def main():
    parser = Loader.parser()
    parser.add_argument("--config")
    args = parser.parse_args()
    for _, config, _, metadata, result in Loader(args):
        # construct command (with inline configuration string if provided)
        cfg = f"genz_icp_data='{args.config}'" if args.config else ''
        cmd = f'{cfg} genz_icp_pipeline {metadata}'
        # use config specified by Loader
        if config is not None: 
            cmd += f' --config {config}'
        # run genz-icp
        subprocess.run(cmd, cwd = result, shell = True)
        # copy trajectory to correct location
        path_to_traj_src, _ = resolve(result / 'results' / 'latest', "poses_tum.txt")
        path_to_traj_dst = result / path_to_traj_src.name
        path_to_traj_dst.write_text(path_to_traj_src.read_text())

if __name__ == '__main__':
    main()