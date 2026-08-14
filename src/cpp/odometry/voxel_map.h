#pragma once

// system
#include <limits>
#include <tbb/blocked_range.h>
#include <optional>
// eigen
#include <Eigen/Dense>
// tsl
#include <ranges>
#include <span>
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
    using point_data_t = PointData;
    using voxel_data_t = VoxelData;
    virtual size_t size() const noexcept = 0;
    virtual std::pair<Eigen::Vector<float, D>&, point_data_t> operator[](size_t) noexcept = 0;
    virtual std::pair<const Eigen::Vector<float, D>&, point_data_t> operator[](size_t) const noexcept = 0;
};

// if F is provided to VoxelMap, it must be derived from Frame
template<size_t D, typename F>
concept FrameType = requires {
    typename F::point_data_t;
    typename F::voxel_data_t;
    requires std::is_base_of_v<Frame<F, D, typename F::point_data_t, typename F::voxel_data_t>, F>;
};

// otherwise, the default FrameType is used
template<size_t D, typename F>
concept DefaultFrameType = std::is_same_v<F, std::span<const Eigen::Vector<float, D>>>;

// indicates that the VoxelMap is well-defined given its template arguments
template<size_t D, typename F>
concept ValidVoxelMap = (D >= 2 && D <= 3) && (DefaultFrameType<D, F> || FrameType<D, F>);

template<size_t D, typename F = std::span<const Eigen::Vector<float, D>>>
requires(ValidVoxelMap<D, F>)
struct VoxelMap {
    explicit VoxelMap(float max_distance, float voxel_size, size_t voxel_capacity) : 
        max_distance_(max_distance), 
        voxel_size_(voxel_size), 
        voxel_resolution_(voxel_size * voxel_size / static_cast<float>(voxel_capacity)), 
        voxel_capacity_(voxel_capacity) { /* STUB */ }
    VoxelMap(const VoxelMap&) = default;
    VoxelMap(VoxelMap&&) = default;
    VoxelMap& operator=(const VoxelMap&) = delete; // NOTE: copy assignment is explicitly deleted to avoid unintended allocations
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
    // transform all points in the map
    void apply(const Transform& T)
    requires(DefaultFrameType<D, F>);
    // update transforms points by pose, extends the map with them, then prunes by max_distance
    void update(F& points, const Transform& pose);
    // extend just adds (a) point(s) to the map directly
    void extend(const F& points);
    void extend(const Eigen::Vector<float, D>& point)
    requires(DefaultFrameType<D, F>);
    // NOTE: this will not protect you if point does not fall into the given voxel
    void extend(const Eigen::Vector<int, D>& voxel, const Eigen::Vector<float, D>& point)
    requires(DefaultFrameType<D, F>);
public:
    template<size_t _D, typename _F>
    struct NeighborInner { /* STUB */ };
    template<size_t _D, typename _F>
    requires(!DefaultFrameType<_D, _F>)
    struct NeighborInner<_D, _F> {
        typename _F::point_data_t data;
        float weight;
    };
    struct Neighbor : NeighborInner<D, F> {
        Eigen::Vector<float, D> point;
        float distance_sq{std::numeric_limits<float>::max()};
    };
    // returns std::nullopt when no correspondence is found
    std::optional<Neighbor> closest(const Eigen::Vector<float, D>& query, float correspondence_distance_sq = std::numeric_limits<float>::max()) const
    requires(DefaultFrameType<D, F>);
    template<typename T = F>
    std::optional<Neighbor> closest(const Eigen::Vector<float, D>& query, typename T::point_data_t data, float correspondence_distance_sq = std::numeric_limits<float>::max()) const
    requires(!DefaultFrameType<D, F>);
private:
    template<size_t _D, typename _F>
    struct VoxelInner { /* STUB */ };
    template<size_t _D, typename _F>
    requires(!DefaultFrameType<_D, _F>)
    struct VoxelInner<_D, _F> {
        friend class VoxelMap<D, F>;
        std::vector<typename _F::point_data_t> point_data;
        typename _F::voxel_data_t data;
    };
    // internal voxel representation
    struct Voxel : VoxelInner<D, F> {
        friend class VoxelMap<D, F>;
    public:
        Voxel(size_t voxel_capacity);
        Voxel(const Voxel&) = default;
        Voxel(Voxel&&) = default;
        Voxel& operator=(const Voxel&) = delete; // NOTE: see voxel_map.h:77
        Voxel& operator=(Voxel&&) = default;
        ~Voxel() = default;
    private:
        void add(const Eigen::Vector<float, D>& point, float voxel_resolution) noexcept
        requires(DefaultFrameType<D, F>);
        template<typename T = F>
        void add(const Eigen::Vector<float, D>& point, typename T::point_data_t data, float voxel_resolution) noexcept
        requires(!DefaultFrameType<D, F>);
    private:
        std::vector<Eigen::Vector<float, D>> points;
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
private:
    struct view_global : public std::ranges::view_interface<view_global> {
        view_global(const VoxelMap<D, F>& map) : map_(&map), begin_(map.map_.cbegin()), end_(map.map_.cend()) { /* STUB */ }
        view_global(view_global& other, tbb::split);
        // helpers
        inline bool empty() const noexcept { return begin_ == end_; }
        inline bool is_divisible() const noexcept { return std::distance(begin_, end_) > 1; }
        // iterator inner class
        struct iterator {
            friend class view_global;
            // proxy for reference type 
            template<typename T = F>
            struct proxy {
                const Eigen::Vector<float, D>& first;
                const typename T::point_data_t& second;
                const proxy* operator->() const noexcept { return this; }
            };
            // iterator description
            using iterator_concept = std::forward_iterator_tag;
            using iterator_category = std::forward_iterator_tag;
            template<typename T = F>
            using value_type = std::conditional_t<DefaultFrameType<D, F>, Eigen::Vector<float, D>, std::pair<Eigen::Vector<float, D>, typename T::point_data_t>>;
            using difference_type = std::ptrdiff_t;
            using reference = std::conditional_t<DefaultFrameType<D, F>, const Eigen::Vector<float, D>&, proxy<>>;
            using pointer = std::conditional_t<DefaultFrameType<D, F>, const Eigen::Vector<float, D>*, proxy<>>;
            // default constructor
            iterator() = default;
            // operators
            reference operator*() const noexcept;
            pointer operator->() const noexcept;
            iterator& operator++();
            iterator operator++(int);
            friend bool operator==(const iterator& lhs, const iterator& rhs) noexcept { return lhs.range_ == rhs.range_ && lhs.i_ == rhs.i_ && lhs.voxel_ == rhs.voxel_; }
            friend bool operator!=(const iterator& lhs, const iterator& rhs) noexcept { return !(lhs == rhs); }
        private:
            iterator(const view_global* range, typename tsl::robin_map<Eigen::Vector<int, D>, Voxel>::const_iterator voxel);
            void satisfy();
            // state
            const view_global* range_{nullptr};
            size_t i_{0};
            typename tsl::robin_map<Eigen::Vector<int, D>, Voxel>::const_iterator voxel_{};
        };
    public:
        inline iterator begin() const { return {this, begin_}; }
        inline iterator end() const { return {this, end_}; }
    private:
        const VoxelMap<D, F>* map_;
        typename tsl::robin_map<Eigen::Vector<int, D>, Voxel>::const_iterator begin_;
        typename tsl::robin_map<Eigen::Vector<int, D>, Voxel>::const_iterator end_;
    };
private:
    struct view_local : public std::ranges::view_interface<view_local> {
        view_local(const VoxelMap<D, F>& map, const Eigen::Vector<float, D>& origin, float radius);
        view_local(view_local& other, tbb::split);
        // helpers
        inline bool empty() const noexcept { return begin_ == end_; }
        inline bool is_divisible() const noexcept { return end_ - begin_ > 1; }
        // iterator inner class
        struct iterator {
            friend class view_local;
            // proxy for the reference type (unable to construct the pair of pointers without getting local copies)
            struct proxy {
                const Eigen::Vector<float, D>& first;
                const typename F::point_data_t& second;
                const proxy* operator->() const noexcept { return this; }
            };
            // iterator description
            using iterator_concept = std::forward_iterator_tag;
            using iterator_category = std::forward_iterator_tag;
            using value_type = std::conditional_t<DefaultFrameType<D, F>, Eigen::Vector<float, D>, std::pair<Eigen::Vector<float, D>, typename F::point_data_t>>;
            using difference_type = std::ptrdiff_t;
            using reference = std::conditional_t<DefaultFrameType<D, F>, const Eigen::Vector<float, D>&, proxy>;
            using pointer = std::conditional_t<DefaultFrameType<D, F>, const Eigen::Vector<float, D>*, proxy>;
            // default constructor (an expectation of forward iterators)
            iterator() = default;
            // operators
            reference operator*() const noexcept;
            pointer operator->() const noexcept;
            iterator& operator++();
            iterator operator++(int);
            friend bool operator==(const iterator& lhs, const iterator& rhs) noexcept { return lhs.range_ == rhs.range_ && lhs.i_ == rhs.i_ && lhs.j_ == rhs.j_; }
            friend bool operator!=(const iterator& lhs, const iterator& rhs) noexcept { return !(lhs == rhs); }
        private:
            iterator(const view_local* range, size_t i);
            void satisfy();
        private:
            const view_local* range_{nullptr};
            size_t i_{0};
            size_t j_{0};
            const Voxel* voxel_{nullptr};
        };
    public:
        inline iterator begin() const { return {this, begin_}; }
        inline iterator end() const { return {this, end_}; }
    private:
        const VoxelMap<D, F>* map_;
        Eigen::Vector<float, D> origin_;
        float radius_sq_;
        Eigen::Vector<int, D> minima_;
        Eigen::Vector<int, D> maxima_;
        size_t begin_{0};
        size_t end_{1};
    };
public:
    view_global view() const {
        return {*this};
    }
    view_local view_within_radius(const Eigen::Vector<float, D>& origin, float radius) const {
        return {*this, origin, radius};
    }
};

#include "odometry/voxel_map.tpp" // IWYU pragma: keep