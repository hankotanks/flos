#pragma once

// sophus
#include <sophus/se3.hpp>
// package
#include "voxel_map.h"
// util
#include "util/cloud.h"

class OdometrySystem {
public:
    explicit OdometrySystem(const std::optional<double>& max_distance = std::nullopt,
        const std::optional<double>& voxel_size = std::nullopt,
        const std::optional<size_t>& voxel_capacity = std::nullopt) {
        if(max_distance) map_.max_distance = *max_distance;
        if(voxel_size) map_.voxel_size = *voxel_size;
        if(voxel_capacity) map_.voxel_capacity = *voxel_capacity;
    }
public:
    OdometrySystem() = delete;
    OdometrySystem(const OdometrySystem&) = delete;
    OdometrySystem(OdometrySystem&&) = delete;
    OdometrySystem& operator=(const OdometrySystem&) = delete;
    OdometrySystem& operator=(OdometrySystem&&) = delete;
    ~OdometrySystem() = default;
public:
    void register_to_map(util::Cloud&& points) noexcept;
private:
    Sophus::SE3f last_pose_;
    Sophus::SE3f last_delta_;
    VoxelMap map_;
};
