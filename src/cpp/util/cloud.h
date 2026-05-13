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
    class Descriptor {
        friend class Cloud;
    public:
        Descriptor() = default;
        Descriptor(const Descriptor&) = default;
        Descriptor(Descriptor&&) = delete;
        Descriptor& operator=(const Descriptor&) = default;
        Descriptor& operator=(Descriptor&&) = delete;
    private:
        void add_field(uint8_t datatype, const std::string& name);
        bool operator==(const Descriptor& other) const noexcept;
        bool operator!=(const Descriptor& other) const noexcept;  
    private:
        std::deque<std::pair<uint8_t, std::string>> attrs_;
        std::map<std::string_view, size_t> names_;
    };
    // forward declaration of Cloud::Point_, which is wrapped by Point and ConstPoint
    template<typename C> class Point_;
public:
    Cloud() = default;
    // if shared is truthy, the fields of the cloud are fixed
    Cloud(const std::shared_ptr<const Descriptor>& desc, bool shared = false) noexcept;
    Cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) noexcept;
    Cloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) noexcept;
    Cloud(const Cloud& other) noexcept;
    Cloud(Cloud&& other) noexcept;
    Cloud& operator=(const Cloud&) = delete; // NOTE: copying large clouds is non-trivial, so it requires explicit construction
    Cloud& operator=(Cloud&& other) noexcept;
public:
    // types representing references to points within the cloud
    // allows accessing its position and field
    using Point = Point_<Cloud>;
    using ConstPoint = Point_<const Cloud>;
    // indexers
    Point operator[](size_t index) { return Point{this, index}; }
    ConstPoint operator[](size_t index) const { return ConstPoint{this, index}; }
public:
    // basic operations
    size_t size() const noexcept;
    bool empty() const noexcept;
    void clear() noexcept;
    void reserve(size_t n) noexcept;
    // get the descriptor
    std::shared_ptr<const Cloud::Descriptor> descriptor() const noexcept;
public:
    // access points
    std::span<const Eigen::Vector3f> points() const noexcept;
    std::span<Eigen::Vector3f> points() noexcept;
    // access a given field by name, throws std::runtime_error if field doesn't exist
    template<typename T>
    std::span<const T> field(const std::string& name) const;
    template<typename T>
    std::span<T> field(const std::string& name);
    // add a new point from another cloud with the same set of scalar fields
    Point add_point(const ConstPoint& point) noexcept;
    // add a new field to the cloud
    // must provide a number of scalars equal to Cloud::size
    // can throw std::runtime_error if
    // - the number of scalars is wrong
    // - the cloud's descriptor is fixed
    template<typename T>
    void add_field(const std::string& name, const std::vector<T>& scalars = {});
    // python bindings
#ifdef GENERATE_CLOUD_PYBIND_MODULE
    // returns a python object containing a `Points` class
    py::object points_as_py_object() const noexcept;
    // returns a python object containing one of `FieldUInt8` `FieldFloat32` etc
    template<typename T>
    py::object field_as_py_object(const std::string& name) const;
    // add a new field to the cloud from a python object containing one of `FieldUInt8` `FieldFloat32` etc
    template<typename T>
    void add_field_from_py_object(const std::string& name, py::object obj);
#endif // GENERATE_CLOUD_PYBIND_MODULE
public:
    // construct a PointCloud2 message from the cloud containing all its scalar fields
    sensor_msgs::msg::PointCloud2 message(const std_msgs::msg::Header& header) const noexcept;
public:
    // base Point_ class from which Cloud::Point and Cloud::ConstPoint derive
    template<typename C>
    class Point_ {
        friend class Cloud;
    public:
        Point_() = delete;
        template<typename C_, typename = std::enable_if_t<std::is_convertible_v<C_*, C*>>>
        Point_(const Point_<C_>& other) : owner_(other.owner_), index_(other.index_) { /* STUB */ }
    public:
        auto& point() const noexcept { return owner_->points_[index_]; }
        template<typename T>
        auto& field(const std::string& name) const { return owner_->template field<T>(name)[index_]; }
    private:
        explicit Point_(C* owner, size_t index) : owner_(owner), index_(index) { /* STUB */ }
        size_t index_;
        C* owner_{nullptr};
    };
private:
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
        void reserve(size_t n) noexcept;
    };
    // allow FieldReader in the implementation access Field
    friend struct FieldReader;
private:
    std::shared_ptr<Descriptor> desc_;
    bool shared_{false};
    std::vector<Eigen::Vector3f> points_;
    std::vector<Field> fields_;
};
}

#ifdef GENERATE_CLOUD_PYBIND_MODULE
// TODO: I would like to automatically generate these bindings somehow
// for every type in sensor_msgs::msg::PointField

using FieldInt8 = std::vector<int8_t>;
using FieldUInt8 = std::vector<uint8_t>;
using FieldInt16 = std::vector<int16_t>;
using FieldUInt16 = std::vector<uint16_t>;
using FieldInt32 = std::vector<int32_t>;
using FieldUInt32 = std::vector<uint32_t>;
using FieldFloat32 = std::vector<float>;
using FieldFloat64 = std::vector<double>;

PYBIND11_MAKE_OPAQUE(FieldInt8);
PYBIND11_MAKE_OPAQUE(FieldUInt8);
PYBIND11_MAKE_OPAQUE(FieldInt16);
PYBIND11_MAKE_OPAQUE(FieldUInt16);
PYBIND11_MAKE_OPAQUE(FieldInt32);
PYBIND11_MAKE_OPAQUE(FieldUInt32);
PYBIND11_MAKE_OPAQUE(FieldFloat32);
PYBIND11_MAKE_OPAQUE(FieldFloat64);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3f>);
#endif // GENERATE_CLOUD_PYBIND_MODULE