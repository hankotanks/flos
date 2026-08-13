#pragma once

// system
#include <limits>
#include <optional>
// eigen
#include <Eigen/Dense>
// tsl
#include <tsl/robin_map.h>
// sophus
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

// represents the data held in each voxel
template<typename Derived, typename PointData>
struct VoxelDataBase;

// represents a single frame of data, which is used to update the VoxelMap
template<typename Derived, size_t D, typename PointData, typename VoxelData>
requires(D >= 2 && D <= 3 && std::is_default_constructible_v<VoxelData> && std::is_base_of_v<VoxelDataBase<Derived, PointData>, VoxelData>)
struct Frame;

// represents the data held in each voxel
template<typename Derived, typename PointData>
struct VoxelDataBase {
    // called for every incoming point during registration
    virtual void update(const PointData& data) noexcept { /* STUB */ }
    // called after a Frame has been added to the map (either through VoxelMap::extend or VoxelMap::update)
    virtual void finish() noexcept { /* STUB */ }
    // used to reject or accept correspondences
    virtual bool accept(const PointData& query) const noexcept { return true; }
    // used to query the voxel with a point's data (the query point's closest neighbor is also supplied)
    virtual float weight(const PointData& query) const noexcept { return 1.f; }
};

// represents a single frame of data, which is used to update the VoxelMap
template<typename Derived, size_t D, typename PointData, typename VoxelData>
requires(D >= 2 && D <= 3 && std::is_default_constructible_v<VoxelData> && std::is_base_of_v<VoxelDataBase<Derived, PointData>, VoxelData>)
struct Frame {
    using derived_t = Derived;
    using point_data_t = PointData;
    using voxel_data_t = VoxelData;
    virtual size_t size() const noexcept = 0;
    virtual std::pair<Eigen::Vector<float, D>&, point_data_t> operator[](size_t) noexcept = 0;
    virtual std::pair<const Eigen::Vector<float, D>&, point_data_t> operator[](size_t) const noexcept = 0;
};

template<>
struct std::hash<Eigen::Vector2i> {
    size_t operator()(const Eigen::Vector2i& voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return (vec[0] * 73856093 ^ vec[1] * 83492791);
    }
};

template<size_t D, typename F>
requires(D >= 2 && D <= 3 && std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>)
struct VoxelMap {
    explicit VoxelMap(float max_distance, float voxel_size, size_t voxel_capacity) : 
        max_distance_(max_distance), 
        voxel_size_(voxel_size), 
        voxel_resolution_(voxel_size * voxel_size / static_cast<float>(voxel_capacity)), 
        voxel_capacity_(voxel_capacity) { /* STUB */ }
    VoxelMap(const VoxelMap&) = delete;
    VoxelMap(VoxelMap&&) = default;
    VoxelMap& operator=(const VoxelMap&) = delete;
    VoxelMap& operator=(VoxelMap&&) = default;
    ~VoxelMap() = default;
public:
    inline Eigen::Vector<int, D> point_to_voxel(const Eigen::Vector<float, D>& point) const noexcept;
    inline Eigen::Vector<float, D> voxel_to_point(const Eigen::Vector<int, D>& voxel) const noexcept;
public:
    inline size_t size() const noexcept { return map_.size(); }
    inline void clear() noexcept { map_.clear(); }
    inline bool empty() const noexcept { return map_.empty(); }
public:
    using Transform = std::conditional_t<D == 2, Sophus::SE2f, Sophus::SE3f>;
    // update transforms points by pose, extends the map with them, then prunes by max_distance
    void update(F& points, const Transform& pose);
    // extend just adds (a) point(s) to the map directly
    void extend(const F& points);
    void extend(const Eigen::Vector<float, D>& point, typename F::point_data_t data);
    // NOTE: this will not protect you if point does not fall into the given voxel
    void extend(const Eigen::Vector<int, D>& voxel, const Eigen::Vector<float, D>& point, typename F::point_data_t data);
    // transform all points in the map
    void apply(const Transform& T);
public:
    // 'algorithms' (parallelized wherever possible)
    template<typename Fn>
    void for_each(Fn&& f) const noexcept;
    template<typename Fn>
    void for_each_within_radius(const Eigen::Vector<float, D>& origin, float radius, Fn&& f) const noexcept;
    template<typename Fn, typename Acc>
    requires requires(Acc a, const Acc& b) { a += b; }
    Acc reduce_within_radius(const Eigen::Vector<float, D>& origin, float radius, Acc init, Fn&& f) const noexcept;
    template<typename Fn>
    auto transform(Fn&& f) const;
public:
    struct Cloud {
        std::vector<Eigen::Vector<float, D>> points;
        std::vector<typename F::point_data_t> data;
    };
    // get all points in the voxel map (currently for debugging)
    const Cloud& cloud() const noexcept;
public:
    struct Neighbor {
        Eigen::Vector<float, D> point;
        typename F::point_data_t data;
        float weight;
        float distance_sq{std::numeric_limits<float>::max()};
    };
    // returns std::nullopt when no correspondence is found
    std::optional<Neighbor> closest(const Eigen::Vector<float, D>& query, typename F::point_data_t data, float correspondence_distance_sq = std::numeric_limits<float>::max()) const;
private:
    // internal voxel representation
    struct Voxel {
        friend class VoxelMap<D, F>;
    public:
        Voxel(size_t voxel_capacity);
        Voxel(const Voxel&) = delete;
        Voxel(Voxel&&) = default;
        Voxel& operator=(const Voxel&) = delete;
        Voxel& operator=(Voxel&&) = default;
        ~Voxel() = default;
    private:
        void add(const Eigen::Vector<float, D>& point, typename F::point_data_t data, float voxel_resolution) noexcept;
    private:
        std::vector<Eigen::Vector<float, D>> points;
        std::vector<typename F::point_data_t> point_data;
        typename F::voxel_data_t data;
    };
    // get lower bound distance for pruning the search space in VoxelMap::closest
    inline float lower_bound_distance_sq(const Eigen::Vector<float, D>& query, const Eigen::Vector<int, D>& voxel) const noexcept;
private:
    // configuration options
    float max_distance_;
    float voxel_size_;
    float voxel_resolution_;
    size_t voxel_capacity_;
    // underlying map structure
    tsl::robin_map<Eigen::Vector<int, D>, Voxel> map_;
};

#include "odometry/voxel_map.tpp" // IWYU pragma: keep