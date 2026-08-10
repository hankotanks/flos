#pragma once

// system
#include <span>
// eigen
#include <Eigen/Core>
// sophus
#include <sophus/se3.hpp>
#include <sophus/se2.hpp>

struct BEV {
public:
    explicit BEV() = default;
    explicit BEV(const Eigen::Vector3f& centroid, const Eigen::Matrix3f& ground, const std::span<const Eigen::Vector3f>& points, const std::span<const uint8_t>& semantics, float max_distance, float voxel_size, size_t voxel_capacity);
    BEV(const BEV&) = default;
    BEV(BEV&&) = default;
    BEV& operator=(const BEV&) = default;
    BEV& operator=(BEV&&) = default;
    ~BEV() = default;
public:
    Sophus::SE2f align(const BEV& other, float cell_size);
    void transform(const Sophus::SE2f& T);
public:
    std::vector<Eigen::Vector2f> projection;
    std::vector<Eigen::Vector2f> centroids;
};