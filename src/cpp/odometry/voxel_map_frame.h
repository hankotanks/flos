#pragma once

// system
#include <limits>
// eigen
#include <Eigen/Dense>

template<typename Derived, typename PointDataType, typename VoxelDataType>
struct Frame;

template<typename Derived, typename PointDataType>
struct Voxel {
    // called for every incoming point during registration
    virtual void update(const PointDataType& data) noexcept { /* STUB */ }
    virtual float weight(const PointDataType& data) const noexcept { return 1.f; }
    virtual void finish() noexcept { /* STUB */ }
};

template<typename Derived, typename PointDataType, typename VoxelDataType>
struct Frame {
    using point_data_type = PointDataType;
    using voxel_data_type = VoxelDataType;
    static_assert(std::is_base_of_v<Voxel<Derived, PointDataType>, VoxelDataType>);
    static_assert(std::is_default_constructible_v<PointDataType>);
    static_assert(std::is_default_constructible_v<VoxelDataType> || std::is_abstract_v<VoxelDataType>);

    virtual size_t size() const noexcept = 0;
    virtual std::pair<Eigen::Vector3f&, point_data_type> operator[](size_t) noexcept = 0;
    virtual std::pair<const Eigen::Vector3f&, point_data_type> operator[](size_t) const noexcept = 0;
};

template<typename F>
struct VoxelData {
    static_assert(std::is_base_of_v<Frame<F, typename F::point_data_type, typename F::voxel_data_type>, F>);
    std::vector<Eigen::Vector3f> points;
    std::vector<typename F::point_data_type> point_data;
    typename F::voxel_data_type data;
public:
    VoxelData(size_t voxel_capacity) {
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
