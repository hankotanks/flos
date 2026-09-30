#!/usr/bin/env python3

import argparse
import pathlib
import numpy as np
import open3d as o3d

def load_labels(path_to_label):
    labels = np.fromfile(path_to_label, dtype = np.uint32)
    return labels

def load_point_cloud(path_to_cloud):
    cloud = o3d.io.read_point_cloud(str(path_to_cloud))
    points = np.asarray(cloud.points)
    if len(points) == 0: raise ValueError(f'empty cloud: {path_to_cloud}')
    return points

def calculate_range_statistics(points):
    ranges = np.linalg.norm(points, axis = 1)
    # TODO: we can add other statistics here if the thesis demands
    # for now I think I only need the maximum range of labelled data
    return {
        'count': len(ranges),
        'max': np.max(ranges),
    }

def process_scene(path_to_scene):
    path_to_clouds = path_to_scene / 'individual_clouds'
    path_to_labels = path_to_scene / 'labels'
    if not path_to_clouds.exists() or not path_to_labels.exists(): return []

    results = []
    for path_to_cloud in sorted(path_to_clouds.glob('*.pcd')):
        path_to_label = path_to_labels / f'{path_to_cloud.stem}.label'
        if not path_to_label.exists(): raise ValueError(f'missing labels: {path_to_label}')

        points = load_point_cloud(path_to_cloud)
        labels = load_labels(path_to_label)
        if len(points) != len(labels): raise ValueError(f'malformed labels: {path_to_label}')

        labelled_mask = labels != 0
        labelled_points = points[labelled_mask]
        if len(labelled_points) == 0: raise ValueError(f'no labelled points in cloud: {path_to_cloud}')

        stats = calculate_range_statistics(labelled_points)
        results.append({ 'scene': path_to_scene.name, 'file': path_to_cloud.name, **stats })

    return results

def output(results):
    if not results: return

    maxs = np.array([r['max'] for r in results])

    print('range of labelled data:')
    print(f'\tavg: {np.mean(maxs):.3f}')
    print(f'\tstd: {np.std(maxs):.3f}')
    print(f'\tmin: {np.min(maxs):.3f}')
    print(f'\tmax: {np.max(maxs):.3f}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type = pathlib.Path,
        help = 'root of digiforests training data')
    args = parser.parse_args()
    if not args.dataset.exists(): raise FileNotFoundError(args.dataset)
    args.dataset = args.dataset / 'raw'
    if not args.dataset.exists(): raise FileNotFoundError(args.dataset)

    results = []
    for split_dir in sorted(args.dataset.iterdir()):
        if not split_dir.is_dir(): continue
        for date_dir in sorted(split_dir.iterdir()):
            if not date_dir.is_dir(): continue
            for scene_dir in sorted(date_dir.iterdir()):
                if not scene_dir.is_dir(): continue
                print(f'processing: {date_dir.name}/{scene_dir.name}')
                scene = process_scene(scene_dir)
                for cloud in scene: cloud['date'] = date_dir.name
                results.extend(scene)

    output(results)

if __name__ == '__main__':
    main()
