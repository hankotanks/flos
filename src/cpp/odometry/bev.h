#pragma once

// system
#include <span>
// eigen
#include <Eigen/Core>
// sophus
#include <sophus/se3.hpp>
#include <sophus/se2.hpp>
// robin_map
#include <tsl/robin_map.h>
// package
#include "odometry/voxel_map.h"

struct BEV {
public:
    explicit BEV(float max_distance, float voxel_size, size_t voxel_capacity) : projection(max_distance, voxel_size, voxel_capacity) { /* STUB */ }
    explicit BEV(const Eigen::Vector3f& centroid, const Eigen::Matrix3f& ground, const std::span<const Eigen::Vector3f>& points, const std::span<const uint8_t>& semantics, float max_distance, float voxel_size, size_t voxel_capacity);
    BEV(const BEV&) = default;
    BEV(BEV&&) = default;
    BEV& operator=(const BEV&) = default;
    BEV& operator=(BEV&&) = default;
    ~BEV() = default;
public:
    std::optional<Sophus::SE2f> align(const BEV& target, size_t max_iterations, float convergence_criterion, float correspondence_distance);
    void transform(const Sophus::SE2f& T);
private:
    using Correspondence = std::tuple<Eigen::Vector2f, Eigen::Vector2f, float>;
    void correspond(const BEV& target, std::vector<Correspondence>& correspondences, float correspondence_distance_sq) const;
public:
    struct Wrapper : public Frame<Wrapper, 2, unsigned char, VoxelDataBase<Wrapper, unsigned char>> {
        explicit Wrapper() = default;
        virtual inline size_t size() const noexcept override final { return points.size(); };
        virtual inline std::pair<Eigen::Vector2f&, unsigned char> operator[](size_t i) noexcept override final { return std::pair<Eigen::Vector2f&, unsigned char>{points[i], 0}; }
        virtual inline std::pair<const Eigen::Vector2f&, unsigned char> operator[](size_t i) const noexcept override final { return std::pair<const Eigen::Vector2f&, unsigned char>{points[i], 0}; }
    public:
        std::vector<Eigen::Vector2f> points;
    };
    VoxelMap<Wrapper, 2> projection;
};