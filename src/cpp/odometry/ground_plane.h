#pragma once

// package
#include "odometry/voxel_map.h"

template<typename F>
class GroundPlane {
public:
    template<typename Predicate>
    explicit GroundPlane(const VoxelMap<F, 3>& map, const Eigen::Vector3f& origin, float max_distance, Predicate&&);
    explicit GroundPlane() = default;
public:
    Eigen::Vector2f project(const Eigen::Vector3f& point) const noexcept;
    Sophus::SE3f lift(const Sophus::SE2f& T) const noexcept;
public:
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f t1 = Eigen::Vector3f::UnitX();
    Eigen::Vector3f t2 = Eigen::Vector3f::UnitY();
private:
    Eigen::Matrix3f plane_ = Eigen::Matrix3f::Identity();
};

#include "odometry/ground_plane.tpp" // IWYU pragma: keep