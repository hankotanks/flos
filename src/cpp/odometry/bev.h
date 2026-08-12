#pragma once

// system
#include <set>
#include <span>
// eigen
#include <Eigen/Core>
// sophus
#include <sophus/se3.hpp>
#include <sophus/se2.hpp>
// robin_map
#include <tsl/robin_map.h>

template<>
struct std::hash<Eigen::Vector2i> {
    size_t operator()(const Eigen::Vector2i& voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return (vec[0] * 73856093 ^ vec[1] * 83492791);
    }
};

struct BEV {
public:
    explicit BEV() = default;
    explicit BEV(const Eigen::Vector3f& centroid, const Eigen::Matrix3f& ground, const std::span<const Eigen::Vector3f>& points, const std::span<const uint8_t>& semantics, float voxel_size, size_t voxel_capacity);
    BEV(const BEV&) = default;
    BEV(BEV&&) = default;
    BEV& operator=(const BEV&) = default;
    BEV& operator=(BEV&&) = default;
    ~BEV() = default;
public:
    std::optional<Sophus::SE2f> align(const BEV& target, size_t max_iterations, float convergence_criterion, float correspondence_distance);
    void transform(const Sophus::SE2f& T);
private:
    void correspondences(const BEV& target, std::vector<size_t>& indices, float correspondence_distance_sq) const;
public:
    std::vector<Eigen::Vector2f> points_projected;
private:
    tsl::robin_map<Eigen::Vector2i, std::set<size_t>> index_;
    float voxel_size_;
};