#pragma once 

#include "odometry/voxel_map.h"

// tbb
#include <tbb/parallel_for_each.h>
#include <tbb/enumerable_thread_specific.h>

template<>
struct std::hash<Eigen::Vector2i> {
    size_t operator()(const Eigen::Vector2i& voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return (vec[0] * 73856093 ^ vec[1] * 83492791);
    }
};

template<>
struct std::hash<Eigen::Vector3i> {
    size_t operator()(const Eigen::Vector3i& voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
    }
};

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
inline Eigen::Vector<int, D> VoxelMap<D, F>::point_to_voxel(const Eigen::Vector<float, D>& point) const noexcept { 
    return (point.array().floor().matrix() / voxel_size_).template cast<int>(); 
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
inline Eigen::Vector<float, D> VoxelMap<D, F>::voxel_to_point(const Eigen::Vector<int, D>& voxel) const noexcept { 
    return (voxel.template cast<float>().array() + 0.5f) * voxel_size_; 
}

// TODO: when moving points out of a voxel, it's important to check if it's now empty
// if it is, it should be removed from the map
template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
template<typename Transform>
requires(TransformEuclid<D, Transform> || TransformAffine<D, Transform>)
void VoxelMap<D, F>::apply(const Transform& T) 
requires(DefaultFrameType<D, F>) {
    std::vector<std::pair<Eigen::Vector<int, D>, Eigen::Vector<float, D>>> extension;
    for(auto it = map_.begin(); it != map_.end(); ++it) {
        const auto& key_before = it.key();
        auto& voxel = it.value();
        static_assert(std::is_same_v<decltype(voxel), Voxel&>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(voxel)>>);
        size_t i_write = 0;
        for(size_t i_read = 0; i_read < voxel.points.size(); ++i_read) {
            Eigen::Vector<float, D> point = T * voxel.points[i_read];
            Eigen::Vector<int, D> key_after = VoxelMap<D, F>::point_to_voxel(point);
            if(key_before != key_after) {
                extension.emplace_back(std::move(key_after), std::move(point));
                continue;
            }
            voxel.points[i_write] = std::move(point);
            ++i_write;
        }
        voxel.points.resize(i_write);
    }
    // add the points which changed voxels
    for(const auto& [voxel, point] : extension) VoxelMap<D, F>::extend(voxel, point);
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
template<typename Transform>
requires(TransformEuclid<D, Transform>)
void VoxelMap<D, F>::update(F& points, const Transform& pose) {
    // transform points by pose
    for(size_t i = 0; i < points.size(); ++i) {
        points[i].first = pose * points[i].first;
    }
    // add points to map
    VoxelMap<D, F>::extend(points);
    // remove points outside of a distance threshold to the current pose
    const Eigen::Vector<float, D>& origin = pose.translation();
    const auto max_distance_sq = max_distance_ * max_distance_;
    for(auto it = map_.begin(); it != map_.end();) {
        const auto& [voxel, inner] = *it;
        if((VoxelMap<D, F>::voxel_to_point(voxel) - origin).squaredNorm() >= max_distance_sq) {
            it = map_.erase(it);
        } else ++it;
    }
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
void VoxelMap<D, F>::extend(const F& points) {
    if constexpr (!DefaultFrameType<D, F>) {
        for(size_t i = 0; i < points.size(); ++i) {
            const auto& [point, data] = points[i];
            const Eigen::Vector<int, D> voxel = VoxelMap<D, F>::point_to_voxel(point);
            auto search = map_.find(voxel);
            if(search == map_.end()) {
                auto [it, _] = map_.try_emplace(voxel, voxel_capacity_);
                search = it;
            }
            // update VoxelData
            search.value().add(point, data, voxel_resolution_);
        }
        for(auto it = map_.begin(); it != map_.end(); ++it) {
            it.value().data.finish();
        }
    } else {
        for(size_t i = 0; i < points.size(); ++i) {
            const auto& point = points[i];
            const Eigen::Vector<int, D> voxel = VoxelMap<D, F>::point_to_voxel(point);
            auto search = map_.find(voxel);
            if(search == map_.end()) {
                auto [it, _] = map_.try_emplace(voxel, voxel_capacity_);
                search = it;
            }
            search.value().add(point, voxel_resolution_);
        }
    }
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
void VoxelMap<D, F>::extend(const Eigen::Vector<float, D>& point) 
requires(DefaultFrameType<D, F>) {
    const Eigen::Vector<int, D> voxel = VoxelMap<D, F>::point_to_voxel(point);
    VoxelMap<D, F>::extend(voxel, point);
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
void VoxelMap<D, F>::extend(const Eigen::Vector<int, D>& voxel, const Eigen::Vector<float, D>& point) 
requires(DefaultFrameType<D, F>) {
    auto search = map_.find(voxel);
    if(search == map_.end()) {
        auto [it, _] = map_.try_emplace(voxel, voxel_capacity_);
        search = it;
    }
    search.value().add(point, voxel_resolution_);
}

#if 0
template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
template<typename Fn>
void VoxelMap<D, F>::for_each(Fn&& f) const noexcept {
    tbb::parallel_for_each(map_.begin(), map_.end(), [&](const auto& entry) {
        const Voxel& voxel = entry.second;
        for(size_t i = 0; i < voxel.points.size(); ++i) {
            if constexpr (!DefaultFrameType<D, F>) {
                std::invoke(f, voxel.points[i], voxel.point_data[i]);
            } else {
                std::invoke(f, voxel.points[i]);
            }
        }
    });
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
template<typename Fn>
void VoxelMap<D, F>::for_each_within_radius(const Eigen::Vector<float, D>& origin, float radius, Fn&& f) const noexcept {
    float radius_sq = radius * radius;
    const Eigen::Vector<float, D> r = Eigen::Vector<float, D>::Constant(radius);
    const Eigen::Vector<int, D> voxel_min = VoxelMap<D, F>::point_to_voxel(origin - r);
    const Eigen::Vector<int, D> voxel_max = VoxelMap<D, F>::point_to_voxel(origin + r);
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
        if((VoxelMap<D, F>::voxel_to_point(voxel) - origin).squaredNorm() > radius_sq) return;
        // otherwise, invoke `f` on all its points
        const Voxel& v = it.value();
        for(size_t j = 0; j < v.points.size(); ++j) {
            if constexpr (!DefaultFrameType<D, F>) {
                std::invoke(f, v.points[i], v.point_data[i]);
            } else {
                std::invoke(f, v.points[i]);
            }
        }
    });
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
template<typename Fn, typename Acc>
requires requires(Acc a, const Acc& b) { a += b; }
Acc VoxelMap<D, F>::reduce_within_radius(const Eigen::Vector<float, D>& origin, float radius, Acc init, Fn&& f) const noexcept {
    tbb::enumerable_thread_specific<Acc> acc_pool;
    if constexpr (!DefaultFrameType<D, F>) {
        VoxelMap<D, F>::for_each_within_radius(origin, radius, [&f, &acc_pool](const Eigen::Vector<float, D>& point, const F::point_data_t& data) {
            auto& acc = acc_pool.local();
            acc = std::invoke(f, acc, point, data);
        });
    } else {
        VoxelMap<D, F>::for_each_within_radius(origin, radius, [&f, &acc_pool](const Eigen::Vector<float, D>& point) {
            auto& acc = acc_pool.local();
            acc = std::invoke(f, acc, point);
        });
    }
    for(const auto& acc : acc_pool) init += acc;
    return init;
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
template<typename Fn>
auto VoxelMap<D, F>::transform(Fn&& f) const {
    if constexpr (!DefaultFrameType<D, F>) {
        using To = std::invoke_result_t<Fn, const Eigen::Vector<float, D>&, const typename F::point_data_t&>;
        tbb::enumerable_thread_specific<std::vector<To>> result_pool;
        VoxelMap<D, F>::for_each([&f, &result_pool](const Eigen::Vector<float, D>& point, const F::point_data_t& data) {
            result_pool.local().emplace_back(std::invoke(f, point, data));
        });
        size_t count = 0;
        for(const auto& result_i : result_pool) count += result_i.size();
        std::vector<To> result;
        result.reserve(count);
        for(auto& result_i : result_pool) {
            result.insert(result.end(), std::make_move_iterator(result_i.begin()), std::make_move_iterator(result_i.end()));
        }
        return result;
    } else {
        using To = std::invoke_result_t<Fn, const Eigen::Vector<float, D>&>;
        tbb::enumerable_thread_specific<std::vector<To>> result_pool;
        VoxelMap<D, F>::for_each([&f, &result_pool](const Eigen::Vector<float, D>& point) {
            result_pool.local().emplace_back(std::invoke(f, point));
        });
        size_t count = 0;
        for(const auto& result_i : result_pool) count += result_i.size();
        std::vector<To> result;
        result.reserve(count);
        for(auto& result_i : result_pool) {
            result.insert(result.end(), std::make_move_iterator(result_i.begin()), std::make_move_iterator(result_i.end()));
        }
        return result;
    }
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
template<typename Fn>
auto VoxelMap<D, F>::transform_filter(Fn&& f) const {
    if constexpr (!DefaultFrameType<D, F>) {
        using Optional = std::invoke_result_t<Fn, const Eigen::Vector<float, D>&, const typename F::point_data_t&>;
        using To = typename std::remove_cvref_t<Optional>::value_type;
        tbb::enumerable_thread_specific<std::vector<To>> result_pool;
        VoxelMap<D, F>::for_each([&f, &result_pool](const Eigen::Vector<float, D>& point, const F::point_data_t& data) {
            auto ret = std::invoke(f, point, data);
            if(ret) result_pool.local().emplace_back(*ret);
        });
        size_t count = 0;
        for(const auto& result_i : result_pool) count += result_i.size();
        std::vector<To> result;
        result.reserve(count);
        for(auto& result_i : result_pool) {
            result.insert(result.end(), std::make_move_iterator(result_i.begin()), std::make_move_iterator(result_i.end()));
        }
        return result;
    } else {
        using Optional = std::invoke_result_t<Fn, const Eigen::Vector<float, D>&>;
        using To = typename std::remove_cvref_t<Optional>::value_type;
        tbb::enumerable_thread_specific<std::vector<To>> result_pool;
        VoxelMap<D, F>::for_each([&f, &result_pool](const Eigen::Vector<float, D>& point) {
            auto ret = std::invoke(f, point);
            if(ret) result_pool.local().emplace_back(*ret);
        });
        size_t count = 0;
        for(const auto& result_i : result_pool) count += result_i.size();
        std::vector<To> result;
        result.reserve(count);
        for(auto& result_i : result_pool) {
            result.insert(result.end(), std::make_move_iterator(result_i.begin()), std::make_move_iterator(result_i.end()));
        }
        return result;
    }
}
#endif

template<size_t D>
requires(D >= 2 && D <= 3)
constexpr size_t offset_pow() {
    size_t n = 1;
    for(size_t i = 0; i < D; ++i) n *= 3;
    return n;
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
std::optional<typename VoxelMap<D, F>::Neighbor> VoxelMap<D, F>::closest(const Eigen::Vector<float, D>& query, float correspondence_distance_sq) const 
requires(DefaultFrameType<D, F>) {
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
    Eigen::Vector<int, D> key = VoxelMap<D, F>::point_to_voxel(query);
    typename VoxelMap<D, F>::Neighbor neighbor;
    neighbor.distance_sq = std::numeric_limits<float>::max();
    const Voxel* voxel{nullptr};
    for(const auto& offset : offsets) {
        const Eigen::Vector<int, D> key_offset = key + offset;
        float lb_distance_sq = VoxelMap<D, F>::lower_bound_distance_sq(query, key_offset);
        if(lb_distance_sq >= neighbor.distance_sq) continue;
        auto search = map_.find(key_offset);
        if(search == map_.end()) continue;
        const Voxel& voxel_shifted = search.value();
        for(size_t i = 0; i < voxel_shifted.points.size(); ++i) {
            float distance_sq = (voxel_shifted.points[i] - query).squaredNorm();
            if(distance_sq >= correspondence_distance_sq) continue;
            if(distance_sq >= neighbor.distance_sq) continue;
            neighbor.point = voxel_shifted.points[i];
            neighbor.distance_sq = distance_sq; 
            voxel = &voxel_shifted;
        }
    }
    if(neighbor.distance_sq < correspondence_distance_sq) {
        return std::make_optional(std::move(neighbor));
    }
    return std::nullopt;
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
template<typename T>
std::optional<typename VoxelMap<D, F>::Neighbor> VoxelMap<D, F>::closest(const Eigen::Vector<float, D>& query, typename T::point_data_t data, float correspondence_distance_sq) const 
requires(!DefaultFrameType<D, F>) {
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
    Eigen::Vector<int, D> key = VoxelMap<D, F>::point_to_voxel(query);
    typename VoxelMap<D, F>::Neighbor neighbor;
    neighbor.distance_sq = std::numeric_limits<float>::max();
    const Voxel* voxel{nullptr};
    for(const auto& offset : offsets) {
        const Eigen::Vector<int, D> key_offset = key + offset;
        float lb_distance_sq = VoxelMap<D, F>::lower_bound_distance_sq(query, key_offset);
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

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
VoxelMap<D, F>::Voxel::Voxel(size_t voxel_capacity) {
    points.reserve(voxel_capacity);
    if constexpr (!DefaultFrameType<D, F>) {
        this->point_data.reserve(voxel_capacity);
    }
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
void VoxelMap<D, F>::Voxel::add(const Eigen::Vector<float, D>& point, float voxel_resolution) noexcept 
requires(DefaultFrameType<D, F>){
    if(points.size() < points.capacity()) {
        size_t flag = std::numeric_limits<size_t>::max();
        for(size_t j = 0; j < points.size(); ++j) {
            if((points[j] - point).squaredNorm() >= voxel_resolution) continue;
            flag = j;
            break;
        }
        if(flag == std::numeric_limits<size_t>::max()) {
            points.emplace_back(point);
        }
    }
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
template<typename T>
void VoxelMap<D, F>::Voxel::add(const Eigen::Vector<float, D>& point, typename T::point_data_t data, float voxel_resolution) noexcept 
requires(!DefaultFrameType<D, F>){
    if(points.size() < points.capacity()) {
        size_t flag = std::numeric_limits<size_t>::max();
        for(size_t j = 0; j < points.size(); ++j) {
            if((points[j] - point).squaredNorm() >= voxel_resolution) continue;
            flag = j;
            break;
        }
        if(flag == std::numeric_limits<size_t>::max()) {
            points.emplace_back(point);
            this->point_data.emplace_back(data);
        }
    }
    this->data.update(data);
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
inline float VoxelMap<D, F>::lower_bound_distance_sq(const Eigen::Vector<float, D>& query, const Eigen::Vector<int, D>& voxel) const noexcept {
    const float h = 0.5f * voxel_size_;
    const Eigen::Vector<float, D> center = VoxelMap<D, F>::voxel_to_point(voxel);
    Eigen::Vector<float, D> d = (query - center).cwiseAbs() - Eigen::Vector<float, D>::Constant(h);
    d = d.cwiseMax(Eigen::Vector<float, D>::Zero());
    return d.squaredNorm();
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
VoxelMap<D, F>::view_global::view_global(VoxelMap<D, F>::view_global& other, tbb::split) : map_(other.map_), begin_(other.begin_), end_(other.end_) {
    const auto distance = std::distance(begin_, end_);
    const auto mid = distance / 2;
    std::advance(begin_, mid);
    other.end_ = begin_;
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
VoxelMap<D, F>::view_global::iterator::reference VoxelMap<D, F>::view_global::iterator::operator*() const noexcept {
    if constexpr (DefaultFrameType<D, F>) {
        return voxel_->second.points[i_];
    } else {
        return {
            voxel_->second.points[i_],
            voxel_->second.point_data[i_]
        };
    }
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
VoxelMap<D, F>::view_global::iterator::pointer VoxelMap<D, F>::view_global::iterator::operator->() const noexcept {
    if constexpr (DefaultFrameType<D, F>) {
        return std::addressof(voxel_->second.points[i_]);
    } else {
        return {
            voxel_->second.points[i_],
            voxel_->second.point_data[i_]
        };
    }
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
VoxelMap<D, F>::view_global::iterator& VoxelMap<D, F>::view_global::iterator::operator++() {
    ++i_;
    satisfy();
    return *this;
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
VoxelMap<D, F>::view_global::iterator VoxelMap<D, F>::view_global::iterator::operator++(int) {
    auto copy = *this;
    ++*this;
    return copy;
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
VoxelMap<D, F>::view_global::iterator::iterator(const VoxelMap<D, F>::view_global* range, typename tsl::robin_map<Eigen::Vector<int, D>, Voxel>::const_iterator voxel) : range_(range), voxel_(voxel) { 
    satisfy(); 
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
void VoxelMap<D, F>::view_global::iterator::satisfy() {
    while(voxel_ != range_->end_) {
        if(i_ < voxel_->second.points.size()) return;
        ++voxel_;
        i_ = 0;
    }
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
VoxelMap<D, F>::view_local::view_local(const VoxelMap<D, F>& map, const Eigen::Vector<float, D>& origin, float radius) : 
    map_(&map), 
    origin_(origin), 
    radius_sq_(radius * radius), 
    minima_(map.point_to_voxel(origin - Eigen::Vector<float, D>::Constant(radius))),
    maxima_(map.point_to_voxel(origin + Eigen::Vector<float, D>::Constant(radius))) {
    for(int d = 0; d < D; ++d) end_ *= static_cast<std::size_t>(maxima_[d] - minima_[d] + 1);
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
VoxelMap<D, F>::view_local::view_local(view_local& other, tbb::split) : 
    map_(other.map_), 
    origin_(other.origin_), 
    radius_sq_(other.radius_sq_), 
    minima_(other.minima_), 
    maxima_(other.maxima_) {
    const auto mid = other.begin_ + (other.end_ - other.begin_) / 2;
    begin_ = mid;
    end_ = other.end_;
    other.end_ = mid;
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
VoxelMap<D, F>::view_local::iterator::reference VoxelMap<D, F>::view_local::iterator::operator*() const noexcept { 
    if constexpr (DefaultFrameType<D, F>) {
        return voxel_->points[j_];
    } else {
        return {
            voxel_->points[j_],
            voxel_->point_data[j_]
        };
    } 
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
VoxelMap<D, F>::view_local::iterator::pointer VoxelMap<D, F>::view_local::iterator::operator->() const noexcept { 
    if constexpr (DefaultFrameType<D, F>) {
        return std::addressof(voxel_->points[j_]);
    } else {
        return operator*().operator->();
    }
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
VoxelMap<D, F>::view_local::iterator& VoxelMap<D, F>::view_local::iterator::operator++() {
    ++j_;
    VoxelMap<D, F>::view_local::iterator::satisfy();
    return *this;
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
VoxelMap<D, F>::view_local::iterator VoxelMap<D, F>::view_local::iterator::operator++(int) {
    auto copy = *this;
    ++(*this);
    return copy;
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
VoxelMap<D, F>::view_local::iterator::iterator(const typename VoxelMap<D, F>::view_local* range, size_t i) : range_(range), i_(i) {
    VoxelMap<D, F>::view_local::iterator::satisfy();
}

template<size_t D, typename F>
requires(ValidVoxelMap<D, F>)
void VoxelMap<D, F>::view_local::iterator::satisfy() {
    while(i_ < range_->end_) {
        if(voxel_ != nullptr) {
            while(j_ < voxel_->points.size()) {
                const Eigen::Vector<float, D>& point = voxel_->points[j_];
                if((point - range_->origin_).squaredNorm() <= range_->radius_sq_) return;
                ++j_;
            }
            j_ = 0;
            voxel_ = nullptr;
        }
        Eigen::Vector<int, D> voxel = range_->minima_;
        for(int d = 0, i = i_; d < D; ++d) {
            const auto extent = static_cast<size_t>(range_->maxima_[d] - range_->minima_[d] + 1);
            voxel[d] += static_cast<int>(i % extent);
            i /= extent;
        }
        ++i_;
        auto it = range_->map_->map_.find(voxel);
        if(it == range_->map_->map_.end()) continue;
        j_ = 0;
        voxel_ = std::addressof(it.value());
    }
    voxel_ = nullptr;
    j_ = 0;
}