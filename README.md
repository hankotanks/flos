<h1 align="center">
  FLOS: Forest LiDAR-only Odometry with Semantics
</h1>

<div align="center">
[![GitHub License](https://img.shields.io/github/license/hankotanks/flos)](/LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/hankotanks/flos)](/)
![ROS2 Kilted](https://img.shields.io/badge/ros%20%7C%20kilted-v0.1.0)
![ROS2 Lyrical](https://img.shields.io/badge/ros%20%7C%20lyrical-v0.1.0)
![ROS2 Rolling](https://img.shields.io/badge/ros%20%7C%20rolling-v0.1.0)
</div>

<p align="center">
  <em>Ongoing subject of my MSc Geodetic Engineering thesis at the University of Bonn</em>
</p>

## About

FLOS achieves pose accuracy comparable to unspecialized LIO methods without an IMU!
Our system employs a lightweight segmentation model based on the [PTv3](https://github.com/Pointcept/PointTransformerV3) architecture,
enabling sensor-rate ground and tree stem detection.
As a prelude to scan-to-map registration, we generate a strong pose hypothesis from these features,
greatly increasing our system's robustness in heavily-vegetated environments.

## Quick Start

For now, our devkit is the fastest way to get started.

```bash
#!/bin/bash
git clone https://github.com/hankotanks/flos && cd flos

# enter the container
source docker/build.sh
source docker/run.sh <path_to_bag> bag
colcon build

# run the mounted bag
ros2 launch flos offline.launch.py path_to_data:=bag
```

The offline system spawns [Rerun](https://rerun.io),
which provides an in-depth visualization of the environment.
