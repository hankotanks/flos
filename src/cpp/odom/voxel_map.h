#pragma once

// system
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

struct VoxelMap {
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
    void update(util::CloudView&& points, const Eigen::Vector3f& origin);
    void update(util::CloudView&& points, const Sophus::SE3f& pose);
    void extend(util::CloudView&& points);
    const util::CloudView& cloud() const noexcept;
    // returns std::nullopt when no correspondence is found
    std::optional<std::pair<util::CloudView::ConstPoint, double>> closest(const util::CloudView::ConstPoint &query) const;
    // a predicate can be supplied taking 2 points and returning a boolean, if truthy, the point is preserved
    using Predicate = std::function<bool(const util::CloudView::ConstPoint& point, const util::CloudView::ConstPoint& query)>;
    std::optional<std::pair<util::CloudView::ConstPoint, double>> closest(const util::CloudView::ConstPoint &query, const Predicate& pred) const;
private:
    // configuration options
    double max_distance_;
    double voxel_size_;
    size_t voxel_capacity_;
private:
    tsl::robin_map<Eigen::Vector3i, util::CloudView> map_;
    std::shared_ptr<const util::CloudView::Descriptor> desc_{nullptr};
};