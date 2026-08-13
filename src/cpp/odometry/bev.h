#pragma once

// eigen
#include <Eigen/Core>
// sophus
#include <sophus/se3.hpp>
#include <sophus/se2.hpp>
// robin_map
#include <tsl/robin_map.h>
// package
#include "odometry/voxel_map.h"

std::optional<Sophus::SE2f> align_projection(VoxelMap<2>& source, const VoxelMap<2>& target, size_t max_iterations, float convergence_criterion, float correspondence_distance);