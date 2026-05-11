#include "voxel_map.h"

namespace {
inline Eigen::Vector3i point_to_voxel(const Eigen::Vector3f& point, double voxel_size) {
    return {
        static_cast<int>(std::floor(point.x() / voxel_size)),
        static_cast<int>(std::floor(point.y() / voxel_size)),
        static_cast<int>(std::floor(point.z() / voxel_size))
    };
}

void voxel_downsample(util::Cloud &cloud, const double voxel_size) {
    tsl::robin_map<Eigen::Vector3i, size_t> grid;
    grid.reserve(cloud.size());
    const std::vector<Eigen::Vector3f>& points = cloud.points();
    for(size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3i voxel = point_to_voxel(points[i], voxel_size);
        if(!grid.contains(voxel)) grid.insert({voxel, i});
    }
    util::Cloud cloud_original{std::move(cloud)};
    std::for_each(grid.cbegin(), grid.cend(), [&](const auto& voxel_and_index) {
        cloud.add_point(cloud_original, voxel_and_index.second);
    });
}

static const std::array<Eigen::Vector3i, 27> voxel_shifts{
    Eigen::Vector3i{ 0,  0,  0},
    Eigen::Vector3i{ 1,  0,  0},
    Eigen::Vector3i{-1,  0,  0},
    Eigen::Vector3i{ 0,  1,  0},
    Eigen::Vector3i{ 0, -1,  0},
    Eigen::Vector3i{ 0,  0,  1},
    Eigen::Vector3i{ 0,  0, -1},
    Eigen::Vector3i{ 1,  1,  0},
    Eigen::Vector3i{ 1, -1,  0},
    Eigen::Vector3i{-1,  1,  0},
    Eigen::Vector3i{-1, -1,  0},
    Eigen::Vector3i{ 1,  0,  1},
    Eigen::Vector3i{ 1,  0, -1},
    Eigen::Vector3i{-1,  0,  1},
    Eigen::Vector3i{-1,  0, -1},
    Eigen::Vector3i{ 0,  1,  1},
    Eigen::Vector3i{ 0,  1, -1},
    Eigen::Vector3i{ 0, -1,  1},
    Eigen::Vector3i{ 0, -1, -1},
    Eigen::Vector3i{ 1,  1,  1},
    Eigen::Vector3i{ 1,  1, -1},
    Eigen::Vector3i{ 1, -1,  1},
    Eigen::Vector3i{ 1, -1, -1},
    Eigen::Vector3i{-1,  1,  1},
    Eigen::Vector3i{-1,  1, -1},
    Eigen::Vector3i{-1, -1,  1},
    Eigen::Vector3i{-1, -1, -1}
};
} // private scope

void VoxelMap::update(util::Cloud&& points, const Eigen::Vector3f& origin) {
    VoxelMap::extend(std::move(points));
    const auto max_distance_sq = max_distance * max_distance;
    for(auto it = map_.begin(); it != map_.end();) {
        const auto& [voxel, inner] = *it;
        if((inner.front().point() - origin).squaredNorm() >= max_distance_sq) {
            it = map_.erase(it);
        } else ++it;
    }
}

void VoxelMap::update(util::Cloud&& points, const Sophus::SE3f& pose) {
    for(size_t i = 0; i < points.size(); ++i) {
        points[i].point() = pose * points[i].point();
    }
    VoxelMap::update(std::move(points), pose.translation());
}

void VoxelMap::extend(util::Cloud&& points) {
    size_t start = cloud_.size();
    if(cloud_.empty()) {
        cloud_ = std::move(points);
    } else {
        for(size_t i = 0; i < points.size(); ++i) {
            cloud_.add_point(points[i]);
        }
    }
    const double map_resolution = std::sqrt(voxel_size * voxel_size / voxel_capacity);
    for(size_t i = start; i < cloud_.size(); ++i) {
        const Eigen::Vector3i& voxel = point_to_voxel(cloud_[i].point(), voxel_size);
        auto search = map_.find(voxel);
        if(search == map_.end()) {
            std::vector<util::Cloud::Point> inner;
            inner.reserve(voxel_capacity);
            inner.emplace_back(cloud_[i]);
            map_.insert({voxel, inner});
        } else {
            auto& inner = search.value();
            if(inner.size() == voxel_capacity) continue;
            if(std::any_of(inner.cbegin(), inner.cend(), [&](const auto& inner_point) { 
                return (inner_point.point() - cloud_[i].point()).norm() < map_resolution; })) continue;
            inner.emplace_back(cloud_[i]);
        }
    }
}

std::optional<std::pair<util::Cloud::Point, double>> VoxelMap::closest(const util::Cloud::Point &query) const {
    const Eigen::Vector3i& voxel = point_to_voxel(query.point(), voxel_size);
    std::optional<std::pair<util::Cloud::Point, double>> closest = std::nullopt;
    std::for_each(voxel_shifts.cbegin(), voxel_shifts.cend(), [&](const auto& voxel_shift) {
        const Eigen::Vector3i& query_voxel = voxel + voxel_shift;
        auto search = map_.find(query_voxel);
        if(search == map_.end()) return;
        const auto& points = search.value();
        const util::Cloud::Point& neighbor = *std::min_element(points.cbegin(), points.cend(), [&](const auto &lhs, const auto &rhs) {
            return (lhs.point() - query.point()).norm() < (rhs.point() - query.point()).norm();
        });
        double distance = (neighbor.point() - query.point()).norm();
        if(!closest || (closest && distance < std::get<1>(*closest))) closest.emplace(neighbor, distance);
    });
    return closest;
}
