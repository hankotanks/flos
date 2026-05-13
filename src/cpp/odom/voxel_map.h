#pragma once

// eigen
#include <Eigen/Dense>
// tsl
#include <tsl/robin_map.h>
// sophus
#include <sophus/se3.hpp>
// util
#include "util/cloud.h"
#include "util/voxel.h" // DO NOT REMOVE, required for the Eigen::Vector3i hash implementation

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
    const util::Cloud& cloud() const noexcept;
    // returns std::nullopt when no correspondence is found
    std::optional<std::pair<util::Cloud::ConstPoint, double>> closest(const util::Cloud::ConstPoint &query) const;
private:
    // configuration options
    double max_distance_;
    double voxel_size_;
    size_t voxel_capacity_;
private:
    tsl::robin_map<Eigen::Vector3i, util::Cloud> map_;
    std::shared_ptr<const util::Cloud::Descriptor> desc_{nullptr};
};