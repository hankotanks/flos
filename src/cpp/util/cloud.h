#pragma once

#include <deque>
#include <map>
#include <variant>
#include <vector>

#include <Eigen/Dense>

#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

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

namespace {
template<typename>
inline constexpr bool False = false;
} // private scope

namespace util {

class Cloud {
public:
    class Descriptor {
        friend class Cloud;
    private:
        Descriptor() = default;
        Descriptor(const Descriptor&) = default;
        Descriptor(Descriptor&&) = delete;
        Descriptor& operator=(const Descriptor&) = default;
        Descriptor& operator=(Descriptor&&) = delete;
    private:
        bool operator==(const Descriptor& other) const noexcept {
            return attrs_ == other.attrs_ && names_ == other.names_;
        }
    private:
        void add_field(uint8_t datatype, const std::string& name) {
            size_t field_count = attrs_.size();
            attrs_.emplace_back(datatype, name);
            const std::string& name_ref = attrs_.back().second;
            names_.emplace(std::string_view(name_ref), field_count);
        }
    private:
        std::deque<std::pair<uint8_t, std::string>> attrs_;
        std::map<std::string_view, size_t> names_;
    };
    
public:
    Cloud(const Descriptor& desc) noexcept;

    Cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) noexcept;

    Cloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) noexcept :
        Cloud(std::const_pointer_cast<const sensor_msgs::msg::PointCloud2>(msg)) { /* STUB */ }

    Cloud(Cloud&& other) noexcept : desc_(other.desc_), points_(std::move(other.points_)), fields_(std::move(other.fields_)) {
        other.points_.clear();
        other.fields_.clear();
        for(size_t i = 0; i < other.fields_.size(); ++i) {
            other.fields_.emplace_back(other.desc_.attrs_[i].first);
        }
    }

    Cloud& operator=(Cloud&& other) noexcept {
        if(this == &other) return *this;
        desc_ = other.desc_;
        points_ = std::move(other.points_);
        fields_ = std::move(other.fields_);
        other.points_.clear();
        other.fields_.clear();
        for(size_t i = 0; i < other.fields_.size(); ++i) {
            other.fields_.emplace_back(other.desc_.attrs_[i].first);
        }
        return *this;
    }

public:
    sensor_msgs::msg::PointCloud2 message(const std_msgs::msg::Header& header) noexcept;

    size_t size() const noexcept {
        return points_.size();
    }

    bool empty() const noexcept {
        return Cloud::size() == 0;
    }

    const std::vector<Eigen::Vector3f>& points() const noexcept {
        return points_;
    }

#ifdef GENERATE_CLOUD_PYBIND_MODULE
    py::object points_as_py_object() const noexcept {
        return py::cast(points_, py::return_value_policy::reference);
    }
#endif // GENERATE_CLOUD_PYBIND_MODULE

    // TODO: better error handling
    template<typename T>
    const std::vector<T>& field(const std::string& name) const {
        try {
            return std::get<std::vector<T>>(fields_[desc_.names_.at(name)].data_);
        } catch(const std::out_of_range& e) {
            throw std::runtime_error("field not found in cloud");
        }
    }

#ifdef GENERATE_CLOUD_PYBIND_MODULE
    template<typename T>
    py::object field_as_py_object(const std::string& name) const {
        return py::cast(Cloud::field<T>(name), py::return_value_policy::reference);
    }
#endif // GENERATE_CLOUD_PYBIND_MODULE

    void add_point(const Cloud& other, size_t i) noexcept;

    template<typename T>
    void add_field(const std::string& name, const std::vector<T>& scalars = {}) noexcept {
        assert(Cloud::size() == scalars.size());

        uint8_t datatype;
        if constexpr (std::is_same_v<T, int8_t>)        datatype = sensor_msgs::msg::PointField::INT8;
        else if constexpr (std::is_same_v<T, uint8_t>)  datatype = sensor_msgs::msg::PointField::UINT8;
        else if constexpr (std::is_same_v<T, int16_t>)  datatype = sensor_msgs::msg::PointField::INT16;
        else if constexpr (std::is_same_v<T, uint16_t>) datatype = sensor_msgs::msg::PointField::UINT16;
        else if constexpr (std::is_same_v<T, int32_t>)  datatype = sensor_msgs::msg::PointField::INT32;
        else if constexpr (std::is_same_v<T, uint32_t>) datatype = sensor_msgs::msg::PointField::UINT32;
        else if constexpr (std::is_same_v<T, float>)    datatype = sensor_msgs::msg::PointField::FLOAT32;
        else if constexpr (std::is_same_v<T, double>)   datatype = sensor_msgs::msg::PointField::FLOAT64;
        else static_assert(False<T>, "unsupported field type");
        
        desc_.add_field(datatype, name);

        fields_.emplace_back(scalars);
    }

#ifdef GENERATE_CLOUD_PYBIND_MODULE
    template<typename T>
    void add_field_from_py_object(const std::string& name, py::object obj) {
        Cloud::add_field(name, obj.cast<std::vector<T>>());
    }
#endif // GENERATE_CLOUD_PYBIND_MODULE

private:
    Cloud(const Cloud&) = delete;
    Cloud& operator=(const Cloud&) = delete;
    
private:
    struct Field {
        template<typename... T>
        using Data_ = std::variant<std::vector<T>...>;
        using Data = Data_<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, float, double>;
        Data data_;

        Field(uint8_t datatype);

        template<typename T>
        Field(const std::vector<T>& scalars) : data_(scalars) { /* STUB */ }
    };

    struct FieldReader {
        template<typename... T>
        using It_ = std::variant<sensor_msgs::PointCloud2ConstIterator<T>...>;
        using It = It_<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, float, double>;
        It it_;

        template<typename T>
        struct Type;
        template<typename T>
        struct Type<sensor_msgs::PointCloud2ConstIterator<T>> { using type = T; };

        FieldReader(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg, uint8_t datatype, const std::string& name);
        void read(Field& field) noexcept;
    };

    struct FieldWriter {
        template<typename... T>
        using It_ = std::variant<sensor_msgs::PointCloud2Iterator<T>...>;
        using It = It_<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, float, double>;
        It it_;

        template<typename T>
        struct Type;
        template<typename T>
        struct Type<sensor_msgs::PointCloud2Iterator<T>> { using type = T; };

        FieldWriter(sensor_msgs::msg::PointCloud2& msg, uint8_t datatype, const std::string& name);

        template<typename T>
        void write(T scalar) noexcept {
            std::visit([scalar](auto&& it) {
                *it = scalar;
                ++it;
            }, it_);
        }
    };
    
    Descriptor desc_;
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