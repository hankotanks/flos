#pragma once

// system
#include <limits>
#include <optional>
// eigen
#include <Eigen/Dense>
// tsl
#include <tsl/robin_map.h>
// sophus
#include <sophus/se3.hpp>

// represents a single frame of data, which is used to update the VoxelMap
template<typename Derived, typename PointData, typename VoxelData>
struct Frame;
// represents the data held in each voxel
// 
template<typename Derived, typename PointData>
struct VoxelData {
    // called for every incoming point during registration
    virtual void update(const PointData& data) noexcept { /* STUB */ }
    // called after a Frame has been added to the map (either through VoxelMap::extend or VoxelMap::update)
    virtual void finish() noexcept { /* STUB */ }
    // used to reject or accept correspondences
    virtual bool accept(const PointData& query) const noexcept { return true; }
    // used to query the voxel with a point's data (the query point's closest neighbor is also supplied)
    virtual float weight(const PointData& query) const noexcept { return 1.f; }
};

template<typename Derived, typename PointData, typename VoxelData>
struct Frame {
    using derived_type = Derived;
    using point_data_type = PointData;
    using voxel_data_type = VoxelData;
    virtual size_t size() const noexcept = 0;
    virtual std::pair<Eigen::Vector3f&, point_data_type> operator[](size_t) noexcept = 0;
    virtual std::pair<const Eigen::Vector3f&, point_data_type> operator[](size_t) const noexcept = 0;
};

template<typename F>
struct VoxelMap {
    static_assert(std::is_default_constructible_v<typename F::voxel_data_type>);
    static_assert(std::is_base_of_v<VoxelData<typename F::derived_type, typename F::point_data_type>, typename F::voxel_data_type>);
public:
    explicit VoxelMap(float max_distance, float voxel_size, size_t voxel_capacity) : 
        max_distance_(max_distance), 
        voxel_size_(voxel_size), 
        voxel_resolution_(voxel_size * voxel_size / static_cast<float>(voxel_capacity)), 
        voxel_capacity_(voxel_capacity) { /* STUB */ }
    VoxelMap(const VoxelMap&) = delete;
    VoxelMap(VoxelMap&&) = delete;
    VoxelMap& operator=(const VoxelMap&) = delete;
    VoxelMap& operator=(VoxelMap&&) = delete;
    ~VoxelMap() = default;
public:
    inline void clear() { map_.clear(); }
    inline bool empty() const { return map_.empty(); }
    // update functions 
    // predicate determines if a point is accepted into the map, with signature:
    // bool(*)(const F::voxel_data_type&, const F::point_data_type&)
    void update(F& points, const Sophus::SE3f& pose);
    void extend(const F& points);
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
        float distance_sq{std::numeric_limits<float>::max()};
    };
    // returns std::nullopt when no correspondence is found
    std::optional<Neighbor> closest(const Eigen::Vector3f& query, typename F::point_data_type data, float max_distance_sq = std::numeric_limits<float>::max()) const;
private:
    // configuration options
    float max_distance_;
    float voxel_size_;
    float voxel_resolution_;
    size_t voxel_capacity_;
private:
    struct Voxel {
        static_assert(std::is_base_of_v<Frame<F, typename F::point_data_type, typename F::voxel_data_type>, F>);
        std::vector<Eigen::Vector3f> points;
        std::vector<typename F::point_data_type> point_data;
        typename F::voxel_data_type data;
    public:
        Voxel(size_t voxel_capacity) {
            points.reserve(voxel_capacity);
            point_data.reserve(voxel_capacity);
        }
    public:
        void add(const Eigen::Vector3f& point, typename F::point_data_type data, float voxel_resolution) noexcept {
            if(points.size() < points.capacity()) {
                size_t flag = std::numeric_limits<size_t>::max();
                for(size_t j = 0; j < points.size(); ++j) {
                    if((points[j] - point).squaredNorm() >= voxel_resolution) continue;
                    flag = j;
                    break;
                }
                if(flag == std::numeric_limits<size_t>::max()) {
                    points.emplace_back(point);
                    point_data.emplace_back(data);
                }
            }
            this->data.update(data);
        }
    };
    tsl::robin_map<Eigen::Vector3i, Voxel> map_;
};

namespace {
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
} // private scope

template<typename F>
void VoxelMap<F>::update(F& points, const Sophus::SE3f& pose) {
    // transform points by pose
    for(size_t i = 0; i < points.size(); ++i) {
        points[i].first = pose * points[i].first;
    }
    // add points to map
    VoxelMap::extend(points);
    // remove points outside of a distance threshold to the current pose
    const Eigen::Vector3f& origin = pose.translation();
    const auto max_distance_sq = max_distance_ * max_distance_;
    for(auto it = map_.begin(); it != map_.end();) {
        const auto& [voxel, inner] = *it;
        if((voxel_to_point(voxel, voxel_size_) - origin).squaredNorm() >= max_distance_sq) {
            it = map_.erase(it);
        } else ++it;
    }
}

template<typename F>
void VoxelMap<F>::extend(const F& points) {
    for(size_t i = 0; i < points.size(); ++i) {
        const auto& [point, data] = points[i];
        const Eigen::Vector3i voxel = point_to_voxel(point, voxel_size_);
        auto search = map_.find(voxel);
        if(search == map_.end()) {
            auto [it, _] = map_.emplace(std::piecewise_construct,
                std::forward_as_tuple(voxel),
                std::forward_as_tuple(voxel_capacity_));
            search = it;
        }
        search.value().add(point, data, voxel_resolution_);
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
    const Eigen::Vector3f center = voxel_to_point(voxel, voxel_size);
    Eigen::Vector3f d = (query - center).cwiseAbs() - Eigen::Vector3f::Constant(h);
    d = d.cwiseMax(Eigen::Vector3f::Zero());
    return d.squaredNorm();
}

template<typename F>
std::optional<typename VoxelMap<F>::Neighbor> VoxelMap<F>::closest(const Eigen::Vector3f& query, typename F::point_data_type data, float max_distance_sq) const {
    // NOTE: moved inside of the function for namespacing reasons 
    // (since VoxelMap::closest must be included in the header due to templating constraints)
    static const std::array<Eigen::Vector3i, 27> offsets{
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
    Eigen::Vector3i key_center = point_to_voxel(query, voxel_size_);
    typename VoxelMap<F>::Neighbor neighbor;
    neighbor.distance_sq = std::numeric_limits<float>::max();
    const Voxel* voxel{nullptr};
    for(const auto& offset : offsets) {
        const Eigen::Vector3i key_offset = key_center + offset;
        float lb_distance_sq = voxel_lower_bound_distance_sq(query, key_offset, voxel_size_);
        if(lb_distance_sq >= neighbor.distance_sq) continue;
        auto search = map_.find(key_offset);
        if(search == map_.end()) continue;
        const Voxel& voxel_shifted = search.value();
        for(size_t i = 0; i < voxel_shifted.points.size(); ++i) {
            float distance_sq = (voxel_shifted.points[i] - query).squaredNorm();
            if(distance_sq >= max_distance_sq) continue;
            if(distance_sq >= neighbor.distance_sq) continue;
            if(voxel_shifted.data.accept(data)) {
                neighbor.point = voxel_shifted.points[i];
                neighbor.data = voxel_shifted.point_data[i];
                neighbor.distance_sq = distance_sq;
                voxel = &voxel_shifted;
            }
        }
    }
    if(neighbor.distance_sq < max_distance_sq) {
        neighbor.weight = voxel->data.weight(data);
        return std::make_optional(std::move(neighbor));
    }
    return std::nullopt;
}