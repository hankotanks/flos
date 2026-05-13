#include "voxel_map.h"

namespace {
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
    const auto max_distance_sq = max_distance_ * max_distance_;
    for(auto it = map_.begin(); it != map_.end();) {
        const auto& [voxel, inner] = *it;
        if((voxel.cast<float>() - origin).squaredNorm() >= max_distance_sq) {
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
    if(desc_ == nullptr) desc_ = points.descriptor();

    const double map_resolution = std::sqrt(voxel_size_ * voxel_size_ / static_cast<double>(voxel_capacity_));
    for(size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3i& voxel = util::point_to_voxel(points[i].point(), voxel_size_);
        auto search = map_.find(voxel);
        if(search == map_.end()) {
            util::Cloud inner{desc_, true};
            inner.reserve(voxel_capacity_);
            inner.add_point(points[i]);
            map_.insert({voxel, std::move(inner)});
        } else {
            auto& inner = search.value();
            if(inner.size() >= voxel_capacity_) continue;
            bool flag = false;
            for(size_t j = 0; j < inner.size(); ++j) {
                if((inner[j].point() - points[i].point()).norm() >= map_resolution) continue;
                flag = true;
                break;
            }
            if(flag) continue;
            inner.add_point(points[i]);
        }
    }
}

const util::Cloud& VoxelMap::cloud() const noexcept {
    static util::Cloud cloud{desc_}; // TODO: evil static cloud to avoid repeated reconstruction
    cloud.clear();
    for(auto it = map_.begin(); it != map_.end();) {
        const auto& [voxel, inner] = *it;
        for(size_t i = 0; i < inner.size(); ++i) {
            cloud.add_point(inner[i]);
        }
    }
    return cloud;
}

std::optional<std::pair<util::Cloud::ConstPoint, double>> VoxelMap::closest(const util::Cloud::ConstPoint &query) const {
    if(map_.empty()) return std::nullopt;
    const Eigen::Vector3i& voxel = util::point_to_voxel(query.point(), voxel_size_);
    std::optional<std::pair<util::Cloud::ConstPoint, double>> closest = std::nullopt;
    std::for_each(voxel_shifts.cbegin(), voxel_shifts.cend(), [&](const auto& voxel_shift) {
        const Eigen::Vector3i& query_voxel = voxel + voxel_shift;
        auto search = map_.find(query_voxel);
        if(search == map_.end()) return;
        const auto& points = search.value();
        util::Cloud::ConstPoint neighbor = points[0];
        for(size_t i = 1; i < points.size(); ++i) {
            if((neighbor.point() - query.point()).norm() < (points[i].point() - query.point()).norm()) continue;
            neighbor = points[i];
        }
        double distance = (neighbor.point() - query.point()).norm();
        if(!closest || (closest && distance < std::get<1>(*closest))) closest.emplace(neighbor, distance);
    });
    return closest;
}
