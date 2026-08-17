#pragma once

#include "odometry/ground_plane.h"
#include "odometry/projection.h"

template<typename Predicate>
void Projection::fit(const VoxelMap<3, SemanticCloud>& map, const Sophus::SE3f& pose, const Sophus::SE3f& T, Predicate&& pred) {
    // update projection
    auto T_icp_projected = ground.project(T);
    target.apply(std::move(T_icp_projected));
    // recompute ground
    if((origin - pose.translation()).squaredNorm() > (max_distance_ground_ * max_distance_ground_ * 0.25f)) {
        origin = pose.translation();
        GroundPlane previous = std::move(ground);
        ground = GroundPlane{map, origin, max_distance_ground_, pred};
        auto T_g2g = ground.project(previous);
        // reproject
        target.apply(T_g2g);
    }
}