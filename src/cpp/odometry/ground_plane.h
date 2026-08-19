#pragma once

// package
#include "odometry/voxel_map.h"
#include "odometry/semantic_cloud.h"

class GroundPlane {
public:
    explicit GroundPlane(float max_distance, float inlier_ratio, float inlier_distance);
    GroundPlane() = delete;
    GroundPlane(const GroundPlane&) = delete;
    GroundPlane(GroundPlane&&) = delete;
    GroundPlane& operator=(const GroundPlane&) = delete;
    GroundPlane& operator=(GroundPlane&&) = delete;
    ~GroundPlane() = default;
public:
    // lift a 2d transform along the planar surface
    Sophus::SE3f lift(const Sophus::SE2f& T) const noexcept;
    // project a point onto the plane, returns (u, v, s) where s is the signed distance to plane
    Eigen::Vector3f project(const Eigen::Vector3f& point) const noexcept;
    // project a transform onto the plane
    std::function<void(Eigen::Vector2f&, float&)> project(const Sophus::SE3f& T) const noexcept;
    // updates the ground plane and returns the transform from the previous plane to the new one
    template<typename Predicate>
    std::function<void(Eigen::Vector2f&, float&)> update(const VoxelMap<3, SemanticCloud>& map, const Eigen::Vector3f& origin, Predicate&&);
private:
    // the desired confidence that our plane fit is correct
    static constexpr float CONFIDENCE{0.99f};
    // RANSAC iteration count
    size_t iterations_;
    // parameters from OdometrySystem::Config
    float max_distance_; // OdometrySystem::Config::max_distance_ground_
    float inlier_distance_; // OdometrySystem::Config::model_iou
private:
    
private:
    Eigen::Vector3f centroid_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f t1_ = Eigen::Vector3f::UnitX();
    Eigen::Vector3f t2_ = Eigen::Vector3f::UnitY();
    Eigen::Vector3f n_ = Eigen::Vector3f::UnitZ();
    Eigen::Matrix3f plane_ = Eigen::Matrix3f::Identity();
};

class BirdsEyeCloud : public Frame<BirdsEyeCloud, 2, float, VoxelDataBase<BirdsEyeCloud, float>> {
public:
    template<typename Predicate>
    explicit BirdsEyeCloud(const SemanticCloud& points, const GroundPlane& ground, Predicate&&);
    explicit BirdsEyeCloud() = default;
    BirdsEyeCloud(const BirdsEyeCloud&) = default;
    BirdsEyeCloud(BirdsEyeCloud&&) = default;
    BirdsEyeCloud& operator=(const BirdsEyeCloud&) = default;
    BirdsEyeCloud& operator=(BirdsEyeCloud&&) = default;
    ~BirdsEyeCloud() = default;
public:
    virtual inline size_t size() const noexcept override final { return points_.size(); };
    virtual std::pair<Eigen::Vector2f, float> operator[](size_t i) const noexcept override final;
private:
    std::vector<Eigen::Vector3f> points_;
};

#include "odometry/ground_plane.tpp" // IWYU pragma: keep