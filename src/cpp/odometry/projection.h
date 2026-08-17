#pragma once

// package
#include "odometry/semantic_cloud.h"
#include "odometry/ground_plane.h"
#include "odometry/voxel_map.h"
#include <iostream>

// grouped for better management of projection state between frames
struct Projection {
    Eigen::Vector3f origin;
    GroundPlane ground;
    VoxelMap<2, BirdsEyeCloud> target;
public:
    Projection(float max_distance, float max_distance_ground, float voxel_size, size_t voxel_capacity) :
        max_distance_(max_distance),
        max_distance_ground_(max_distance_ground),
        voxel_size_(voxel_size * 2.f),
        voxel_capacity_(voxel_capacity * 2),
        target(max_distance, voxel_size * 2.f, voxel_capacity * 2) { /* STUB */ }
public:
    // aligns the current projection to the previous
    std::optional<Sophus::SE2f> align(const BirdsEyeCloud& source, size_t max_iterations, float convergence_criterion, float correspondence_distance);
    // update ground plane and maintain state
    template<typename Predicate>
    void fit(const VoxelMap<3, SemanticCloud>& map, const SemanticCloud& points, const Sophus::SE3f& pose, const Sophus::SE3f& T, Predicate&& pred);
private:
    float max_distance_;
    float max_distance_ground_;
    float voxel_size_;
    size_t voxel_capacity_;
};

template<typename Predicate>
void Projection::fit(const VoxelMap<3, SemanticCloud>& map, const SemanticCloud& points, const Sophus::SE3f& pose, const Sophus::SE3f& T, Predicate&& pred) {
    if((origin - pose.translation()).squaredNorm() > (max_distance_ground_ * max_distance_ground_ * 0.0625f)) {
        origin = pose.translation();
        GroundPlane ground_new{map, origin, max_distance_ground_, pred};
        auto T_g2g = ground_new.project(ground);
        // reproject
        target.apply(T_g2g);
        ground = ground_new;
    }
    auto T_icp_projected = ground.project(T);
    target.apply(std::move(T_icp_projected));
    // target.clear();
    // BirdsEyeCloud bev{points, ground, [this](const Semantics& data) {
    //     return data.prediction == 3;
    // }};
    // target.extend(bev);
    // update current frame projection
}