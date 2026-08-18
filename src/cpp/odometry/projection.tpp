#pragma once

// package
#include "odometry/ground_plane.h"
#include "odometry/projection.h"

template<typename Predicate>
void Projection::fit(const VoxelMap<3, SemanticCloud>& map, const Sophus::SE3f& pose, const Sophus::SE3f& T, Predicate&& pred) {
    // update projection
    auto T_icp_projected = ground_.project(T);
    target_.apply(std::move(T_icp_projected));
    // recompute ground
    if((origin_ - pose.translation()).squaredNorm() > (max_distance_ground_ * max_distance_ground_ * 0.25f)) {
        origin_ = pose.translation();
        GroundPlane previous = std::move(ground_);
        ground_ = GroundPlane{map, origin_, max_distance_ground_, pred};
        auto T_g2g = ground_.project(previous);
        // reproject
        target_.apply(T_g2g);
    }
}