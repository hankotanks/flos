#pragma once

// package
#include "odometry/voxel_map.h"
#include "odometry/semantic_cloud.h"


class GroundPlane;

struct BirdsEyeCloud : public Frame<BirdsEyeCloud, 2, float, VoxelDataBase<BirdsEyeCloud, float>> {
    template<typename Predicate>
    BirdsEyeCloud(const SemanticCloud& points, const GroundPlane& ground, Predicate&&);
    virtual inline size_t size() const noexcept override final { return points_.size(); };
    virtual std::pair<Eigen::Vector2f, float> operator[](size_t i) const noexcept override final;
private:
    std::vector<Eigen::Vector3f> points_;
};

class GroundPlane {
public:
    template<typename Predicate>
    explicit GroundPlane(const VoxelMap<3, SemanticCloud>& map, const Eigen::Vector3f& origin, float max_distance, Predicate&&);
    explicit GroundPlane() = default;
public:
    // lift a 2d transform along the planar surface
    Sophus::SE3f lift(const Sophus::SE2f& T) const noexcept;
    // project a transform onto the plane
    // returns a lambda with signature (const Eigen::Vector2f&) -> Eigen::Vector2f
    std::function<void(Eigen::Vector2f&, float&)> project(const Sophus::SE3f& T) const noexcept;
    std::function<void(Eigen::Vector2f&, float&)> project(const GroundPlane& source) const noexcept;
    // project a point onto the plane
    Eigen::Vector3f project(const Eigen::Vector3f& point) const noexcept;
public:
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f t1 = Eigen::Vector3f::UnitX();
    Eigen::Vector3f t2 = Eigen::Vector3f::UnitY();
    Eigen::Vector3f n = Eigen::Vector3f::UnitZ();
private:
    Eigen::Matrix3f plane_ = Eigen::Matrix3f::Identity();
};

#include "odometry/ground_plane.tpp" // IWYU pragma: keep