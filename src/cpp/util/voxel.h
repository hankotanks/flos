#pragma once

// eigen
#include <Eigen/Dense>

template <>
struct std::hash<Eigen::Vector3i> {
    std::size_t operator()(const Eigen::Vector3i& voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
    }
};

namespace util {
inline Eigen::Vector3i point_to_voxel(const Eigen::Vector3f& point, double voxel_size) {
    return {
        static_cast<int>(std::floor(point.x() / voxel_size)),
        static_cast<int>(std::floor(point.y() / voxel_size)),
        static_cast<int>(std::floor(point.z() / voxel_size))
    };
}

inline Eigen::Vector3f voxel_to_point(const Eigen::Vector3i& voxel, double voxel_size) {
    return {
        (static_cast<float>(voxel.x()) + 0.5f) * static_cast<float>(voxel_size),
        (static_cast<float>(voxel.y()) + 0.5f) * static_cast<float>(voxel_size),
        (static_cast<float>(voxel.z()) + 0.5f) * static_cast<float>(voxel_size)
    };
}
}