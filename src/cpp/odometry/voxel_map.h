#pragma once

// system
#include <any>
#include <functional>
// eigen
#include <Eigen/Dense>
// tsl
#include <tsl/robin_map.h>
// sophus
#include <sophus/se3.hpp>
// util
#include "util/cloud.h"
#include "util/voxel.h" // DO NOT REMOVE, required for the Eigen::Vector3i hash implementation

template<typename Derived, typename Data>
struct Points;

template<typename Derived>
struct VoxelData {
    // called for every incoming point during registration
    virtual void update(const Derived& points, size_t i) noexcept = 0;
};

template<typename Derived, typename Data>
struct Points {
    using data_type = Data;
    static_assert(std::is_base_of_v<VoxelData<Derived>, Data>);
    static_assert(std::is_default_constructible_v<Data> || std::is_abstract_v<Data>);

    virtual size_t size() const noexcept = 0;
    virtual util::CloudView::Point operator[](size_t) noexcept = 0;
    virtual util::CloudView::ConstPoint operator[](size_t) const noexcept = 0;
};

template<typename P>
struct VoxelMap {
    static_assert(std::is_base_of_v<Points<P, typename P::data_type>, P>);
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
    void update(P& points, const Eigen::Vector3f& origin);
    void update(P& points, const Sophus::SE3f& pose);
    void extend(P& points);
    // get all points in the voxel map (currently for debugging)
    const util::CloudView& cloud() const noexcept;
    // returns std::nullopt when no correspondence is found
    std::optional<std::pair<util::CloudView::ConstPoint, double>> closest(const util::CloudView::ConstPoint& query) const;
    // a predicate can be supplied taking 2 points and returning a boolean, if truthy, the point is preserved
    using Predicate = std::function<bool(const util::CloudView::ConstPoint& point, const util::CloudView::ConstPoint& query)>;
    std::optional<std::pair<util::CloudView::ConstPoint, double>> closest(const util::CloudView::ConstPoint& query, const Predicate& pred) const;
private:
    struct Voxel {
        util::CloudView points;
        typename P::data_type data;
        Voxel(const std::shared_ptr<const util::CloudView::Descriptor>& desc, size_t voxel_capacity);
        void add_point(double resolution, const P& other, size_t i) noexcept;
    };
private:
    // configuration options
    double max_distance_;
    double voxel_size_;
    size_t voxel_capacity_;
private:
    tsl::robin_map<Eigen::Vector3i, Voxel> map_;
    std::shared_ptr<const util::CloudView::Descriptor> desc_{nullptr};
};

template<typename P>
VoxelMap<P>::Voxel::Voxel(const std::shared_ptr<const util::CloudView::Descriptor>& desc, size_t voxel_capacity) : points(desc) {
    points.reserve(voxel_capacity);
}

template<typename P>
void VoxelMap<P>::Voxel::add_point(double resolution, const P& other, size_t i) noexcept {
    if(points.size() < points.capacity()) {
        bool flag = false;
        for(size_t j = 0; j < points.size(); ++j) {
            if((points[j].point() - other[i].point()).norm() >= resolution) continue;
            flag = true;
            break;
        }
        if(!flag) {
            points.add_point(other[i]);
        }
    }
    data.update(other, i);
}

template<typename P>
void VoxelMap<P>::update(P& points, const Eigen::Vector3f& origin) {
    VoxelMap::extend(points);
    const auto max_distance_sq = max_distance_ * max_distance_;
    for(auto it = map_.begin(); it != map_.end();) {
        const auto& [voxel, inner] = *it;
        if((voxel.template cast<float>() - origin).squaredNorm() >= max_distance_sq) {
            it = map_.erase(it);
        } else ++it;
    }
}

template<typename P>
void VoxelMap<P>::update(P& points, const Sophus::SE3f& pose) {
    for(size_t i = 0; i < points.size(); ++i) {
        points[i].point() = pose * points[i].point();
    }
    VoxelMap::update(points, pose.translation());
}

template<typename P>
void VoxelMap<P>::extend(P& points) {
    const double resolution = std::sqrt(voxel_size_ * voxel_size_ / static_cast<double>(voxel_capacity_));
    for(size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3i& voxel = util::point_to_voxel(points[i].point(), voxel_size_);
        auto search = map_.find(voxel);
        if(search == map_.end()) {
            auto [it, _] = map_.emplace(std::piecewise_construct,
                std::forward_as_tuple(voxel),
                std::forward_as_tuple(desc_, voxel_capacity_));
            search = it;
        }
        search.value().add_point(resolution, points, i);
    }
}

template<typename P>
const util::CloudView& VoxelMap<P>::cloud() const noexcept {
    assert(desc_);
    static util::CloudView cloud{desc_};
    cloud.clear();
    cloud.reserve(map_.size() * voxel_capacity_);
    for(auto it = map_.begin(); it != map_.end(); ++it) {
        const auto& [_, voxel] = *it;
        for(size_t i = 0; i < voxel.points.size(); ++i) {
            cloud.add_point(voxel.points[i]);
        }
    }
    return cloud;
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

// TODO: this can be implemented as a specialization of the predicate version of the function
// I'm keeping it as is for now for diagnostic purposes
template<typename P>
std::optional<std::pair<util::CloudView::ConstPoint, double>> VoxelMap<P>::closest(const util::CloudView::ConstPoint& query) const {
    if(map_.empty()) return std::nullopt;
    const Eigen::Vector3i& voxel = util::point_to_voxel(query.point(), voxel_size_);
    std::optional<std::pair<util::CloudView::ConstPoint, double>> closest = std::nullopt;
    std::for_each(voxel_shifts.cbegin(), voxel_shifts.cend(), [&](const auto& voxel_shift) {
        const Eigen::Vector3i& query_voxel = voxel + voxel_shift;
        auto search = map_.find(query_voxel);
        if(search == map_.end()) return;
        const Voxel& voxel = search.value();
        util::CloudView::ConstPoint neighbor = voxel.points[0];
        for(size_t i = 1; i < voxel.points.size(); ++i) {
            if((neighbor.point() - query.point()).norm() < (voxel.points[i].point() - query.point()).norm()) continue;
            neighbor = voxel.points[i];
        }
        double distance = (neighbor.point() - query.point()).norm();
        if(!closest || (closest && distance < std::get<1>(*closest))) closest.emplace(neighbor, distance);
    });
    return closest;
}

template<typename P>
std::optional<std::pair<util::CloudView::ConstPoint, double>> VoxelMap<P>::closest(const util::CloudView::ConstPoint& query, const VoxelMap::Predicate& pred) const {
    if(map_.empty()) return std::nullopt;
    const Eigen::Vector3i& voxel = util::point_to_voxel(query.point(), voxel_size_);
    std::optional<std::pair<util::CloudView::ConstPoint, double>> closest = std::nullopt;
    std::for_each(voxel_shifts.cbegin(), voxel_shifts.cend(), [&](const auto& voxel_shift) {
        const Eigen::Vector3i& query_voxel = voxel + voxel_shift;
        auto search = map_.find(query_voxel);
        if(search == map_.end()) return;
        const Voxel& voxel = search.value();
        std::optional<util::CloudView::ConstPoint> neighbor;
        for(size_t i = 0; i < voxel.points.size(); ++i) {
            if(!(pred)(voxel.points[i], query)) continue;
            if(!neighbor || ((neighbor->point() - query.point()).norm() >= (voxel.points[i].point() - query.point()).norm())) {
                neighbor = voxel.points[i];
            }
        }
        if(neighbor) {
            double distance = (neighbor->point() - query.point()).norm();
            if(!closest || (closest && distance < std::get<1>(*closest))) closest.emplace(*neighbor, distance);
        }
    });
    return closest;
}