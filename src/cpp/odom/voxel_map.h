#pragma once

// eigen
#include <Eigen/Dense>
// tsl
#include <tsl/robin_map.h>
// sophus
#include <sophus/se3.hpp>
// util
#include "util/cloud.h"

template <>
struct std::hash<Eigen::Vector3i> {
    std::size_t operator()(const Eigen::Vector3i& voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
    }
};

struct VoxelMap {
    explicit VoxelMap(double max_distance, double voxel_size, size_t voxel_capacity) : 
        max_distance_(max_distance), voxel_size_(voxel_size), voxel_capacity_(voxel_capacity) { /* STUB */ }
    VoxelMap(const VoxelMap&) = delete;
    VoxelMap(VoxelMap&&) = delete;
    VoxelMap& operator=(const VoxelMap&) = delete;
    VoxelMap& operator=(VoxelMap&&) = delete;
    ~VoxelMap() = default;
public:
    inline void clear() { map_.clear(); }
    inline bool empty() const { return map_.empty(); }
    void update(util::Cloud&& points, const Eigen::Vector3f& origin);
    void update(util::Cloud&& points, const Sophus::SE3f& pose);
    void extend(util::Cloud&& points);
    std::optional<std::pair<util::Cloud::Point, double>> closest(const util::Cloud::Point &query) const;
    inline const util::Cloud& cloud() const noexcept { return cloud_; }
private:
    // configuration options
    double max_distance_;
    double voxel_size_;
    size_t voxel_capacity_;
private:
    tsl::robin_map<Eigen::Vector3i, std::vector<util::Cloud::Point>> map_;
    util::Cloud cloud_;

};