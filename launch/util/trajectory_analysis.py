#!/usr/bin/env python3

import argparse
import copy
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from evo.core.trajectory import PoseTrajectory3D

def convert_gt_to_tum(path_raw: Path) -> Path:
    path_tum = path_raw.with_suffix(".tum")

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

def load_tum(path: Path) -> PoseTrajectory3D:
    data = np.loadtxt(path)
    if data.ndim == 1: data = data.reshape(1, -1)
    return PoseTrajectory3D(
        timestamps = data[:, 0],
        positions_xyz = data[:, 1:4],
        orientations_quat_wxyz = np.column_stack([data[:, 7], data[:, 4], data[:, 5], data[:, 6]]))

def crop(traj: PoseTrajectory3D, t_start: float, t_end: float) -> PoseTrajectory3D:
    mask = np.logical_and(traj.timestamps >= t_start, traj.timestamps <= t_end)
    return PoseTrajectory3D(
        timestamps = traj.timestamps[mask],
        positions_xyz = traj.positions_xyz[mask],
        orientations_quat_wxyz = traj.orientations_quat_wxyz[mask])

def resample(traj: PoseTrajectory3D, stamps: np.ndarray) -> PoseTrajectory3D:
    t_min = traj.timestamps[0]
    t_max = traj.timestamps[-1]

    mask = (stamps >= t_min) & (stamps <= t_max)
    stamps = stamps[mask]
    
    positions_xyz_interp = scipy.interpolate.interp1d(traj.timestamps, traj.positions_xyz, 
        axis = 0, kind = "linear", bounds_error = True)
    positions_xyz = positions_xyz_interp(stamps)

    quat_wxyz = traj.orientations_quat_wxyz
    quat_xyzw = np.column_stack([quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]])

    rotations = scipy.spatial.transform.Rotation.from_quat(quat_xyzw)
    rotations_slerp = scipy.spatial.transform.Slerp(traj.timestamps, rotations)
    rotations = rotations_slerp(stamps)

    quat_xyzw = rotations.as_quat()
    quat_wxyz = np.column_stack([
        quat_xyzw[:, 3],
        quat_xyzw[:, 0],
        quat_xyzw[:, 1],
        quat_xyzw[:, 2],
    ])

    return PoseTrajectory3D(
        timestamps = stamps,
        positions_xyz = positions_xyz,
        orientations_quat_wxyz = quat_wxyz)

def plot_traj(ax, traj, label, is_ground_truth = False):
    ax.plot(traj.positions_xyz[:, 0], traj.positions_xyz[:, 1], traj.positions_xyz[:, 2], 
        label = label,
        linewidth = 3 if is_ground_truth else 1)

def main(path_gt: Path, path_out: Path):
    path_gt = convert_gt_to_tum(path_gt)
    traj_gt = load_tum(path_gt)

    traj = []
    path = [p for p in path_out.iterdir() if p.is_file() and p.suffix == ".tum" and p != path_gt]
    for path_i_tum in path:
        path_i_tum = Path(path_i_tum)
        traj.append((path_i_tum, load_tum(path_i_tum)))
    if not traj: raise RuntimeError("no trajectories provided")

    t_start = max(traj.timestamps[0] for _, traj in traj)
    t_end = min(traj.timestamps[-1] for _, traj in traj)
    if t_start > t_end: raise RuntimeError(f"no overlapping interval: {t_start} > {t_end}")
    print(f"common interval: {t_start:.3f} -> {t_end:.3f}")

    traj_gt = crop(traj_gt, t_start, t_end)
    if len(traj_gt.timestamps) < 2:
        raise RuntimeError("ground truth is too short after cropping")

    ax = plt.figure(figsize = (12, 10)).add_subplot(111, projection="3d")

    plot_traj(ax, traj_gt, "ground_truth", True)

    for path_i, traj_i in traj:
        traj_i_cropped = crop(traj_i, t_start, t_end)
        if len(traj_i_cropped.timestamps) < 2:
            print(f"skipping {path_i.name}: cropping left only {len(traj_i_cropped.timestamps)} samples")
            continue

        traj_i_aligned = copy.deepcopy(resample(traj_i_cropped, traj_gt.timestamps))
        try:
            traj_i_aligned.align(traj_gt, correct_scale = True)
        except Exception as e:
            print(f"skipping {path_i.name}: alignment failed: {e}")
            continue

        plot_traj(ax, traj_i_aligned, path_i.stem)

    """
    for path_i, traj_i in traj:
        traj_i_cropped = crop(traj_i, t_start, t_end)

        if len(traj_i_cropped.timestamps) < 2:
            print(f"skipping {path_i.name}: cropping left only {len(traj_i_cropped.timestamps)} samples")
            continue

        # --- resample onto GT timestamps ---
        traj_i_resampled = resample(traj_i_cropped, traj_gt.timestamps)

        # --- enforce identical shape for evo.align() ---
        n = min(len(traj_i_resampled.timestamps), len(traj_gt.timestamps))

        traj_i_resampled = PoseTrajectory3D(
            timestamps=traj_i_resampled.timestamps[:n],
            positions_xyz=traj_i_resampled.positions_xyz[:n],
            orientations_quat_wxyz=traj_i_resampled.orientations_quat_wxyz[:n],
        )

        traj_gt_aligned = PoseTrajectory3D(
            timestamps=traj_gt.timestamps[:n],
            positions_xyz=traj_gt.positions_xyz[:n],
            orientations_quat_wxyz=traj_gt.orientations_quat_wxyz[:n],
        )

        try:
            traj_i_resampled.align(traj_gt_aligned, correct_scale=True)
        except Exception as e:
            print(f"skipping {path_i.name}: alignment failed: {e}")
            continue

        plot_traj(ax, traj_i_resampled, path_i.stem)
    """

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.tight_layout()
    plt.savefig(path_out / "out.png", dpi = 300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "visualize trajectories produced by RecorderNode")
    parser.add_argument("path_gt", 
        help = "path to ground truth file from Digiforests (pose.txt)")
    parser.add_argument("path_out",
        help = "path to folder produced by RecorderNode")

    args = parser.parse_args()

    path_gt = Path(args.path_gt)
    path_out = Path(args.path_out)
    main(path_gt, path_out)