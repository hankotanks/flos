#pragma once

// eigen
#include <Eigen/Dense>
// tsl
#include <limits>
#include <optional>
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
    // using Predicate = std::function<bool(const typename F::voxel_data_type& data, typename F::point_data_type point, typename F::point_data_type query)>;
    // contains the point, its data, and its distance to the query point
    struct Neighbor {
        Eigen::Vector3f point;
        typename F::point_data_type data;
        float weight;
        float distance_sq;
        explicit Neighbor() = default;
    };
    // returns std::nullopt when no correspondence is found
    template<typename Predicate>
    std::optional<Neighbor> closest(const Eigen::Vector3f& query, typename F::point_data_type data, Predicate&& pred, float max_distance_sq = std::numeric_limits<float>::max()) const;
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
        if((util::voxel_to_point(voxel, voxel_size_) - origin).squaredNorm() >= max_distance_sq) {
            it = map_.erase(it);
        } else ++it;
    }
}

template<typename F>
void VoxelMap<F>::extend(const F& points, const Sophus::SE3f& pose) {
    const double resolution_sq = voxel_size_ * voxel_size_ / static_cast<double>(voxel_capacity_);
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
        search.value().add(resolution_sq, point, data);
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

inline float voxel_lower_bound_distance_sq(const Eigen::Vector3f& query, const Eigen::Vector3i& voxel, float voxel_size) {
    const float h = 0.5f * voxel_size;
    const Eigen::Vector3f center = util::voxel_to_point(voxel, voxel_size);
    Eigen::Vector3f d = (query - center).cwiseAbs() - Eigen::Vector3f::Constant(h);
    d = d.cwiseMax(Eigen::Vector3f::Zero());
    return d.squaredNorm();
}

template<typename F>
template<typename Predicate>
std::optional<typename VoxelMap<F>::Neighbor> VoxelMap<F>::closest(const Eigen::Vector3f& query, typename F::point_data_type data, Predicate&& pred, float correspondence_distance_sq) const {
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
    Eigen::Vector3i center_voxel = util::point_to_voxel(query, voxel_size_);
    const Eigen::Vector3f base_origin = util::voxel_to_point(center_voxel, voxel_size_);
    typename VoxelMap<F>::Neighbor closest;
    closest.distance_sq = std::numeric_limits<float>::max();
    bool closest_found = false;
    for(const auto& voxel_shift : voxel_shifts) {
        const Eigen::Vector3i query_voxel = center_voxel + voxel_shift;
        float lb = voxel_lower_bound_distance_sq(query, query_voxel, voxel_size_);
        if(lb >= closest.distance_sq) continue;
        auto search = map_.find(query_voxel);
        if(search == map_.end()) continue;
        const VoxelData<F>& voxel = search.value();
        size_t neighbor = 0;
        float neighbor_distance_sq = std::numeric_limits<float>::max();
        bool neighbor_found = false;
        for(size_t i = 0; i < voxel.points.size(); ++i) {
            float distance_sq = (voxel.points[i] - query).squaredNorm();
            if(distance_sq >= correspondence_distance_sq) continue;
            if(distance_sq >= closest.distance_sq) continue;
            if(distance_sq >= neighbor_distance_sq) continue;
            if(!(pred)(voxel.data, voxel.point_data[i], data)) continue;
            neighbor = i;
            neighbor_distance_sq = distance_sq;
            neighbor_found = true;
        }
        if(neighbor_found && neighbor_distance_sq < closest.distance_sq) {
            closest_found = true;
            closest.point = voxel.points[neighbor];
            closest.data = voxel.point_data[neighbor];
            closest.weight = voxel.data.weight(data);
            closest.distance_sq = neighbor_distance_sq;
        }
    }
    return closest_found ? std::make_optional(closest) : std::nullopt;
}