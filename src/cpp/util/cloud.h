#pragma once

// system
#include <deque>
#include <map>
#include <span>
#include <variant>
#include <vector>
// eigen
#include <Eigen/Dense>
// ros
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
// pybind
#ifdef GENERATE_CLOUD_PYBIND_MODULE
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/eigen.h>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
namespace py = pybind11;
#endif // GENERATE_CLOUD_PYBIND_MODULE

namespace util {
class Cloud {
public:
    explicit Cloud() = default;
    Cloud(std::span<const Eigen::Vector3f> points) noexcept;
    Cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) noexcept;
    Cloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) noexcept;
    Cloud(const Cloud& other) noexcept = default;
    Cloud(Cloud&& other) noexcept;
    Cloud& operator=(const Cloud&) noexcept = delete; // NOTE: copying large clouds is non-trivial, so it requires explicit construction
    Cloud& operator=(Cloud&& other) noexcept;
public:
    // basic operations
    inline size_t size() const noexcept { return points_.size(); }
    inline size_t capacity() const noexcept { return points_.capacity(); }
    inline bool empty() const noexcept { return points_.empty(); }
public:
    void clear() noexcept;
    void resize(size_t count);
    void reserve(size_t new_cap);
public:
    const Eigen::Vector3f& operator[](size_t i) const { return points_[i]; }
    Eigen::Vector3f& operator[](size_t i) { return points_[i]; }
public:
    // access points
    std::span<const Eigen::Vector3f> points() const noexcept;
    std::span<Eigen::Vector3f> points() noexcept;
    // access a given field by name, throws std::runtime_error if field doesn't exist
    template<typename T>
    std::span<const T> field(const std::string& name) const;
    template<typename T>
    std::span<T> field(const std::string& name);
    // add a new field to the cloud
    // must provide a number of scalars equal to Cloud::size
    // can throw std::runtime_error if the number of scalars is wrong
    template<typename T>
    void add_field(const std::string& name, const std::vector<T>& scalars = {}) noexcept;
    template<typename T>
    void add_field(const std::string& name, const T& scalar) noexcept;
public:
    // python bindings
#ifdef GENERATE_CLOUD_PYBIND_MODULE
    // returns a python object containing a `Points` class
    py::object points_as_py_object() const noexcept;
    // returns a python object containing one of `FieldUInt8` `FieldFloat32` etc
    template<typename T>
    py::object field_as_py_object(const std::string& name) const;
    // add a new field to the cloud from a python object containing one of `FieldUInt8` `FieldFloat32` etc
    template<typename T>
    void add_field_from_py_object(const std::string& name, py::object obj) noexcept;
#endif // GENERATE_CLOUD_PYBIND_MODULE
public:
    void downsample(float voxel_size);
    // filter the cloud based on a certain field
    template<typename T, typename Predicate>
    void filter(const std::string& name, Predicate&&);
    // filter by distance
    void filter_by_distance(const Eigen::Vector3f& origin, float minima, float maxima);
    // filter by distance and scalar predicate
    // applies the distance filter to all points where the predicate returns truthy
    template<typename T, typename Predicate>
    void filter_by_distance(const Eigen::Vector3f& origin, float minima, float maxima, const std::string& name, Predicate&&);
    // sort the field based on a comparator over a field
    template<typename T, typename Comparator> 
    void sort(const std::string& name, Comparator&&);
    // apply to transform to each element of a field
    template<typename T, typename Transform>
    void transform(const std::string& name, Transform&&);
public:
    // construct a PointCloud2 message from the cloud containing all its scalar fields
    sensor_msgs::msg::PointCloud2 message(const std_msgs::msg::Header& header) const noexcept;
protected:
    // C++ shenanigans to prevent static_assert from being optimized out of std::visit calls
    template<typename>
    inline static const constexpr bool False = false;
    // this is a wrapper around all supported field types
    struct Field {
        template<typename... T>
        using Data_ = std::variant<std::vector<T>...>;
        using Data = Data_<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, float, double>;
        Data data_;
        Field(uint8_t datatype);
        template<typename T>
        Field(const std::vector<T>& scalars) : data_(scalars) { /* STUB */ }
        void clear() noexcept;
        void reserve(size_t new_cap);
        void resize(size_t count);
        template<typename T>
        T& operator[](size_t index);
        template<typename T>
        const T& operator[](size_t index) const;
    };
    // allow FieldReader in the implementation access Field
    friend struct FieldReader;
protected:
    class Descriptor {
        friend class Cloud;
    public:
        Descriptor() = default;
        Descriptor(const Descriptor&);
        Descriptor(Descriptor&&) = delete;
        Descriptor& operator=(const Descriptor&);
        Descriptor& operator=(Descriptor&&) = delete;
    private:
        void add_field(uint8_t datatype, const std::string& name);
        bool operator==(const Descriptor& other) const noexcept;
        bool operator!=(const Descriptor& other) const noexcept;  
    private:
        std::deque<std::pair<uint8_t, std::string>> attrs_;
        std::map<std::string_view, size_t> names_;
    };
protected:
    Descriptor desc_;
    std::vector<Eigen::Vector3f> points_;
    std::vector<Field> fields_;
};
}

#include "util/cloud.tpp" // IWYU pragma: keep