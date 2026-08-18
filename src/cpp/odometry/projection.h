#pragma once

// package
#include "odometry/semantic_cloud.h"
#include "odometry/ground_plane.h"
#include "odometry/voxel_map.h"

// grouped for better management of projection state between frames
class Projection {
public:
    Projection(float max_distance, float max_distance_ground, float voxel_size, size_t voxel_capacity) :
        max_distance_ground_(max_distance_ground),
        target_(max_distance, voxel_size * 2.f, voxel_capacity * 2),
        source_(max_distance, voxel_size * 2.f, voxel_capacity * 2) { /* STUB */ }
    Projection(const Projection&) = delete;
    Projection(Projection&&) = delete;
    Projection& operator=(const Projection&) = delete;
    Projection& operator=(Projection&&) = delete;
    ~Projection() = default;
public:
    // aligns the current projection to the previous
    std::optional<Sophus::SE2f> align(const BirdsEyeCloud& source, size_t max_iterations, float convergence_criterion);
    // update ground plane and maintain state
    template<typename Predicate>
    void fit(const VoxelMap<3, SemanticCloud>& map, const Sophus::SE3f& pose, const Sophus::SE3f& T, Predicate&& pred);
public:
    Eigen::Vector3f origin_;
    GroundPlane ground_;
    VoxelMap<2, BirdsEyeCloud> target_;
private:
    float max_distance_ground_;
    VoxelMap<2, BirdsEyeCloud> source_;
};

#include "odometry/projection.tpp" // IWYU pragma: keep