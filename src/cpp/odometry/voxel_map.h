#pragma once

// system
#include <functional>
// eigen
#include <Eigen/Dense>
// tsl
#include <tsl/robin_map.h>
// sophus
#include <sophus/se3.hpp>
// package
#include "frame.h"
// util
#include "util/voxel.h" // DO NOT REMOVE, required for the Eigen::Vector3i hash implementation

template<typename F>
struct VoxelMap {
public:
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
    // update functions
    void update(F& points, const Sophus::SE3f& pose);
    void extend(const F& points, const Sophus::SE3f& pose);
public:
    struct Cloud {
        std::vector<Eigen::Vector3f> points;
        std::vector<typename F::point_data_type> data;
    };
    // get all points in the voxel map (currently for debugging)
    const Cloud& cloud() const noexcept;
public:
    // a predicate can be supplied taking 2 points' data and returning a boolean, if truthy, the match is preserved
    using Predicate = std::function<bool(const typename F::voxel_data_type& data, typename F::point_data_type point, typename F::point_data_type query)>;
    // contains the point, its data, and its distance to the query point
    struct Neighbor {
        const Eigen::Vector3f point;
        typename F::point_data_type data;
        float weight;
        float distance;
    };
    // returns std::nullopt when no correspondence is found
    std::optional<Neighbor> closest(const Eigen::Vector3f& query, typename F::point_data_type data, const Predicate& pred) const;
private:
    // configuration options
    double max_distance_;
    double voxel_size_;
    size_t voxel_capacity_;
private:
    tsl::robin_map<Eigen::Vector3i, VoxelData<F>> map_;
};

template<typename F>
void VoxelMap<F>::update(F& points, const Sophus::SE3f& pose) {
    // transform points by pose
    for(size_t i = 0; i < points.size(); ++i) {
        points[i].first = pose * points[i].first;
    }
    // add points to map
    VoxelMap::extend(points, pose);
    // remove points outside of a distance threshold to the current pose
    const Eigen::Vector3f& origin = pose.translation();
    const auto max_distance_sq = max_distance_ * max_distance_;
    for(auto it = map_.begin(); it != map_.end();) {
        const auto& [voxel, inner] = *it;
        if((voxel.template cast<float>() - origin).squaredNorm() >= max_distance_sq) {
            it = map_.erase(it);
        } else ++it;
    }
}

template<typename F>
void VoxelMap<F>::extend(const F& points, const Sophus::SE3f& pose) {
    const double resolution = std::sqrt(voxel_size_ * voxel_size_ / static_cast<double>(voxel_capacity_));
    for(size_t i = 0; i < points.size(); ++i) {
        const auto& [point, data] = points[i];
        const Eigen::Vector3i voxel = util::point_to_voxel(point, voxel_size_);
        auto search = map_.find(voxel);
        if(search == map_.end()) {
            auto [it, _] = map_.emplace(std::piecewise_construct,
                std::forward_as_tuple(voxel),
                std::forward_as_tuple(voxel_capacity_));
            search = it;
        }
        search.value().add(resolution, point, data);
    }
    // update the VoxelData
    for(auto it = map_.begin(); it != map_.end(); ++it) {
        it.value().data.finish();
    }
}

template<typename F>
const VoxelMap<F>::Cloud& VoxelMap<F>::cloud() const noexcept {
    static VoxelMap<F>::Cloud cloud;
    cloud.points.clear();
    cloud.data.clear();
    cloud.points.reserve(map_.size() * voxel_capacity_);
    cloud.data.reserve(map_.size() * voxel_capacity_);
    for(auto it = map_.begin(); it != map_.end(); ++it) {
        const auto& [_, voxel] = *it;
        cloud.points.insert(cloud.points.end(), voxel.points.cbegin(), voxel.points.cend());
        cloud.data.insert(cloud.data.end(), voxel.point_data.cbegin(), voxel.point_data.cend());
    }
    return cloud;
}

template<typename F>
std::optional<typename VoxelMap<F>::Neighbor> VoxelMap<F>::closest(const Eigen::Vector3f& query, typename F::point_data_type data, const VoxelMap<F>::Predicate& pred) const {
    // NOTE: moved inside of the function for namespacing reasons 
    // (since VoxelMap::closest must be included in the header due to templating constraints)
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
    // find the closest match that respects the provided predicate
    if(map_.empty()) return std::nullopt;
    Eigen::Vector3i voxel = util::point_to_voxel(query, voxel_size_);
    std::optional<typename VoxelMap<F>::Neighbor> closest = std::nullopt;
    std::for_each(voxel_shifts.cbegin(), voxel_shifts.cend(), [&](const auto& voxel_shift) {
        const Eigen::Vector3i& query_voxel = voxel + voxel_shift;
        auto search = map_.find(query_voxel);
        if(search == map_.end()) return;
        const VoxelData<F>& voxel = search.value();
        std::optional<std::pair<const Eigen::Vector3f, typename F::point_data_type>> neighbor;
        for(size_t i = 0; i < voxel.points.size(); ++i) {
            if(!(pred)(voxel.data, voxel.point_data[i], data)) continue;
            if(!neighbor || ((neighbor->first - query).norm() >= (voxel.points[i] - query).norm())) {
                neighbor.emplace(voxel.points[i], voxel.point_data[i]);
            }
        }
        if(neighbor) {
            float weight = voxel.data.weight(data);
            float distance = (neighbor->first - query).norm();
            if(!closest || (closest && distance < closest->distance)) {
                closest.emplace(neighbor->first, neighbor->second, weight, distance);
            }
        }
    });
    return closest;
}