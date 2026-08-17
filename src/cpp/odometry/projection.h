#pragma once

// package
#include "odometry/semantic_cloud.h"
#include "odometry/ground_plane.h"
#include "odometry/voxel_map.h"

// util::Cloud wrapper for fast access of fields
struct BirdsEyeCloud : public Frame<BirdsEyeCloud, 2, float, VoxelDataBase<BirdsEyeCloud, float>> {
    friend Belief;
    template<typename Predicate>
    BirdsEyeCloud(const SemanticCloud& points, const GroundPlane<SemanticCloud>& ground, Predicate&&);
    virtual inline size_t size() const noexcept override final { return points_.size(); };
    virtual std::pair<Eigen::Vector2f, float> operator[](size_t i) const noexcept override final;
private:
    std::vector<Eigen::Vector3f> points_;
};

// grouped for better management of projection state between frames
struct Projection {
    Eigen::Vector3f origin;
    GroundPlane<SemanticCloud> ground;
    VoxelMap<2> prev;
    VoxelMap<2> curr;
public:
    Projection(float max_distance, float voxel_size, size_t voxel_capacity) :
        prev(max_distance, voxel_size * 2.f, voxel_capacity * 2),
        curr(max_distance, voxel_size * 2.f, voxel_capacity * 2) { /* STUB */ }
public:
    // aligns the current projection to the previous
    std::optional<Sophus::SE2f> align(size_t max_iterations, float convergence_criterion, float correspondence_distance);
};
