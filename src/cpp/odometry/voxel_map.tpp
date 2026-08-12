#pragma once 

#include "odometry/voxel_map.h"

template<typename F, size_t D>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
inline Eigen::Vector<int, D> VoxelMap<F, D>::point_to_voxel(const Eigen::Vector<float, D>& point) const noexcept { 
    return (point.array().floor().matrix() / voxel_size_).template cast<int>(); 
}

template<typename F, size_t D>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
inline Eigen::Vector<float, D> VoxelMap<F, D>::voxel_to_point(const Eigen::Vector<int, D>& voxel) const noexcept { 
    return voxel.template cast<float>().array() + 0.5f; 
}

template<typename F, size_t D>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
void VoxelMap<F, D>::update(F& points, const Pose& pose) {
    // transform points by pose
    for(size_t i = 0; i < points.size(); ++i) {
        points[i].first = pose * points[i].first;
    }
    // add points to map
    VoxelMap<F, D>::extend(points);
    // remove points outside of a distance threshold to the current pose
    const Eigen::Vector<float, D>& origin = pose.translation();
    const auto max_distance_sq = max_distance_ * max_distance_;
    for(auto it = map_.begin(); it != map_.end();) {
        const auto& [voxel, inner] = *it;
        if((VoxelMap<F, D>::voxel_to_point(voxel) - origin).squaredNorm() >= max_distance_sq) {
            it = map_.erase(it);
        } else ++it;
    }
}

template<typename F, size_t D>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
void VoxelMap<F, D>::extend(const F& points) {
    for(size_t i = 0; i < points.size(); ++i) {
        const auto& [point, data] = points[i];
        VoxelMap<F, D>::extend(point, data);
    }
    // update the VoxelData
    for(auto it = map_.begin(); it != map_.end(); ++it) {
        it.value().data.finish();
    }
}

template<typename F, size_t D>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
void VoxelMap<F, D>::extend(const Eigen::Vector<float, D>& point, typename F::point_data_t data) {
    const Eigen::Vector<int, D> voxel = VoxelMap<F, D>::point_to_voxel(point);
    VoxelMap<F, D>::extend(voxel, point, data);
}

template<typename F, size_t D>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
void VoxelMap<F, D>::extend(const Eigen::Vector<int, D>& voxel, const Eigen::Vector<float, D>& point, typename F::point_data_t data) {
    auto search = map_.find(voxel);
    if(search == map_.end()) {
        auto [it, _] = map_.emplace(std::piecewise_construct,
            std::forward_as_tuple(voxel),
            std::forward_as_tuple(voxel_capacity_));
        search = it;
    }
    search.value().add(point, data, voxel_resolution_);
}

template<typename F, size_t D>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
void VoxelMap<F, D>::transform(const Pose& T) {
    std::vector<std::tuple<Eigen::Vector<int, D>, Eigen::Vector<float, D>, typename F::point_data_t>> extension;
    for(auto it = map_.begin(); it != map_.end(); ++it) {
        const auto& key_before = it.key();
        auto& voxel = it.value();
        static_assert(std::is_same_v<decltype(voxel), Voxel&>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(voxel)>>);
        size_t i_write = 0;
        for(size_t i_read = 0; i_read < voxel.points.size(); ++i_read) {
            Eigen::Vector<float, D> point = T * voxel.points[i_read];
            Eigen::Vector<int, D> key_after = VoxelMap<F, D>::point_to_voxel(point);
            if(key_before != key_after) {
                extension.emplace_back(std::move(key_after), std::move(point), std::move(voxel.point_data[i_read]));
                continue;
            }
            voxel.points[i_write] = std::move(point);
            if(i_read != i_write) voxel.point_data[i_write] = std::move(voxel.point_data[i_read]);
            ++i_write;
        }
        voxel.points.resize(i_write);
        voxel.point_data.resize(i_write);
    }
    // add the points which changed voxels
    for(const auto& [voxel, point, data] : extension) VoxelMap<F, D>::extend(voxel, point, data);
}

template<typename F, size_t D>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
template<typename Fn>
void VoxelMap<F, D>::for_each(Fn&& f) const noexcept {
    tbb::parallel_for_each(map_.begin(), map_.end(), [&](const auto& entry) {
        const Voxel& voxel = entry.second;
        for(size_t i = 0; i < voxel.points.size(); ++i) {
            std::invoke(f, voxel.points[i], voxel.point_data[i]);
        }
    });
}

template<typename F, size_t D>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
template<typename Fn>
void VoxelMap<F, D>::for_each_in_radius(const Eigen::Vector<float, D>& origin, float radius, Fn&& f) const noexcept {
    float radius_sq = radius * radius;
    const Eigen::Vector<float, D> r = Eigen::Vector<float, D>::Constant(radius);
    const Eigen::Vector<int, D> voxel_min = VoxelMap<F, D>::point_to_voxel(origin - r);
    const Eigen::Vector<int, D> voxel_max = VoxelMap<F, D>::point_to_voxel(origin + r);
    // find the 'local' space that will be iterated over
    const Eigen::Vector<size_t, D> extent = (voxel_max - voxel_min).template cast<size_t>() + Eigen::Vector<size_t, D>::Constant(1);
    size_t voxel_count = 1;
    for(size_t d = 0; d < D; ++d) voxel_count *= extent[d];
    // dispatch over linear indices...
    tbb::parallel_for(size_t{0}, voxel_count, [&](size_t i) {
        // ... then recover the voxel key
        Eigen::Vector<int, D> voxel = voxel_min;
        for(size_t d = 0; d < D; ++d) {
            voxel[d] += static_cast<int>(i % extent[d]);
            i /= extent[d];
        }
        // check if voxel exists
        auto it = map_.find(voxel);
        if(it == map_.end()) return;
        // if the voxel is too far away, skip it
        if((VoxelMap<F, D>::voxel_to_point(voxel) - origin).squaredNorm() > radius_sq) return;
        // otherwise, invoke `f` on all its points
        const Voxel& v = it.value();
        for(size_t j = 0; j < v.points.size(); ++j) std::invoke(f, v.points[j], v.point_data[j]);
    });
}

template<typename F, size_t D>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
const VoxelMap<F, D>::Cloud& VoxelMap<F, D>::cloud() const noexcept {
    static VoxelMap<F, D>::Cloud cloud;
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

template<size_t D>
requires(D >= 2 && D <= 3)
constexpr size_t offset_pow() {
    size_t n = 1;
    for(size_t i = 0; i < D; ++i) n *= 3;
    return n;
}

template<typename F, size_t D>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
std::optional<typename VoxelMap<F, D>::Neighbor> VoxelMap<F, D>::closest(const Eigen::Vector<float, D>& query, typename F::point_data_t data, float correspondence_distance_sq) const {
    // NOTE: moved inside of the function for namespacing reasons 
    // (since VoxelMap::closest must be included in the header due to templating constraints)
    static const auto offsets = std::invoke([]() {
        std::array<Eigen::Vector<int, D>, offset_pow<D>()> result{};
        for(size_t i = 0; i < result.size(); ++i) {
            size_t n = i;
            for(size_t d = 0; d < D; ++d) {
                result[i][d] = static_cast<int>(n % 3) - 1;
                n /= 3;
            }
        }
        return result;
    });
    // find the closest match that respects the provided predicate
    if(map_.empty()) return std::nullopt;
    Eigen::Vector<int, D> key = VoxelMap<F, D>::point_to_voxel(query);
    typename VoxelMap<F, D>::Neighbor neighbor;
    neighbor.distance_sq = std::numeric_limits<float>::max();
    const Voxel* voxel{nullptr};
    for(const auto& offset : offsets) {
        const Eigen::Vector<int, D> key_offset = key + offset;
        float lb_distance_sq = VoxelMap<F, D>::lower_bound_distance_sq(query, key_offset);
        if(lb_distance_sq >= neighbor.distance_sq) continue;
        auto search = map_.find(key_offset);
        if(search == map_.end()) continue;
        const Voxel& voxel_shifted = search.value();
        for(size_t i = 0; i < voxel_shifted.points.size(); ++i) {
            float distance_sq = (voxel_shifted.points[i] - query).squaredNorm();
            if(distance_sq >= correspondence_distance_sq) continue;
            if(distance_sq >= neighbor.distance_sq) continue;
            if(voxel_shifted.data.accept(data)) {
                neighbor.point = voxel_shifted.points[i];
                neighbor.data = voxel_shifted.point_data[i];
                neighbor.distance_sq = distance_sq;
                voxel = &voxel_shifted;
            }
        }
    }
    if(neighbor.distance_sq < correspondence_distance_sq) {
        neighbor.weight = voxel->data.weight(data);
        return std::make_optional(std::move(neighbor));
    }
    return std::nullopt;
}

template<typename F, size_t D>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
VoxelMap<F, D>::Voxel::Voxel(size_t voxel_capacity) {
    points.reserve(voxel_capacity);
    point_data.reserve(voxel_capacity);
}

template<typename F, size_t D>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
void VoxelMap<F, D>::Voxel::add(const Eigen::Vector<float, D>& point, typename F::point_data_t data, float voxel_resolution) noexcept {
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

template<typename F, size_t D>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
inline float VoxelMap<F, D>::lower_bound_distance_sq(const Eigen::Vector<float, D>& query, const Eigen::Vector<int, D>& voxel) const noexcept {
    const float h = 0.5f * voxel_size_;
    const Eigen::Vector<float, D> center = VoxelMap<F, D>::voxel_to_point(voxel);
    Eigen::Vector<float, D> d = (query - center).cwiseAbs() - Eigen::Vector<float, D>::Constant(h);
    d = d.cwiseMax(Eigen::Vector<float, D>::Zero());
    return d.squaredNorm();
}