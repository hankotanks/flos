#pragma once

// package
#include "odometry/semantic_cloud.h"
#include "odometry/ground_plane.h"
#include "odometry/voxel_map.h"

// grouped for better management of projection state between frames
class Projection {
    friend class OfflineNode;
public:
    template<typename GroundPredicate, typename TargetPredicate>
    explicit Projection(float max_distance, float max_distance_ground, float voxel_size, float inlier_ratio, float inlier_distance, GroundPredicate&& pred_ground, TargetPredicate&& pred_target) :
        max_distance_(max_distance),
        max_distance_ground_(max_distance_ground),
        voxel_size_(voxel_size * Projection::VOXEL_SIZE_SCALAR),
        inlier_distance_(inlier_distance),
        pred_ground_(pred_ground),
        pred_target_(pred_target),
        ground_(max_distance_ground_, inlier_ratio, inlier_distance_),
        target_(max_distance_, voxel_size_),
        source_(max_distance_, voxel_size_) { /* STUB */ }
    Projection(const Projection&) = delete;
    Projection(Projection&&) = delete;
    Projection& operator=(const Projection&) = delete;
    Projection& operator=(Projection&&) = delete;
    ~Projection() = default;
public:
    // aligns the current projection to the previous
    std::optional<Sophus::SE3f> align(const SemanticCloud& source, size_t max_iterations, float convergence_criterion);
    // update ground plane and maintain state
    void fit(const VoxelMap<3, SemanticCloud>& map, const Sophus::SE3f& pose, const Sophus::SE3f& T);
private:
    // NOTE: we used a fixed voxel size for projected maps
    // because VoxelMap::closest only checks neighboring voxels, and in the refinement stage, 
    // we need to be VERY sure that any large pose errors are corrected
    // static constexpr float CORRESPONDENCE_DISTANCE{2.f};
    static constexpr float VOXEL_SIZE_SCALAR{1.41421f}; // sqrt(2)
    // shared parameters with OdometrySystem
    float max_distance_;
    float max_distance_ground_;
    float voxel_size_;
    float inlier_distance_;
private:
    std::function<bool(const Semantics&)> pred_ground_;
    std::function<bool(const Semantics&)> pred_target_;
private:
    Eigen::Vector3f origin_;
    GroundPlane ground_;
    VoxelMap<2, BirdsEyeCloud> target_;
    VoxelMap<2, BirdsEyeCloud> source_;
};