#include "util/cloud.h"

// system
#include <memory>
#include <set>
#include <regex>
#include <stdexcept>
// robin_map
#include <tsl/robin_map.h>

using i1 = int8_t;
using u1 = uint8_t;
using i2 = int16_t;
using u2 = uint16_t;
using i4 = int32_t;
using u4 = uint32_t;
using f4 = float;
using f8 = double;

namespace {
inline std::string fix_frame_id(const std::string &frame_id) {
    return std::regex_replace(frame_id, std::regex("^/"), "");
}

inline Eigen::Vector3i point_to_voxel(const Eigen::Vector3f& point, double voxel_size) {
    return {
        static_cast<int>(std::floor(point.x() / voxel_size)),
        static_cast<int>(std::floor(point.y() / voxel_size)),
        static_cast<int>(std::floor(point.z() / voxel_size))
    };
}
} // private scope

namespace util {
Cloud::Descriptor::Descriptor(const Descriptor& other) : attrs_(other.attrs_) {
    for(size_t i = 0; i < attrs_.size(); ++i) {
        names_.emplace(std::string_view(attrs_[i].second), i);
    }
}

Cloud::Descriptor& Cloud::Descriptor::operator=(const Descriptor& other) {
    if(this == &other) return *this;
    attrs_ = other.attrs_;
    names_.clear();
    for(size_t i = 0; i < attrs_.size(); ++i) {
        names_.emplace(std::string_view(attrs_[i].second), i);
    }
    return *this;
}

void Cloud::Descriptor::add_field(uint8_t datatype, const std::string& name) {
    size_t field_count = attrs_.size();
    attrs_.emplace_back(datatype, name);
    const std::string& name_ref = attrs_.back().second;
    names_.emplace(std::string_view(name_ref), field_count);
}

bool Cloud::Descriptor::operator==(const Descriptor& other) const noexcept {
    return attrs_ == other.attrs_ && names_ == other.names_;
}

bool Cloud::Descriptor::operator!=(const Descriptor& other) const noexcept {
    return !(*this == other);
}

Cloud::Cloud(std::span<const Eigen::Vector3f> points) noexcept : Cloud::Cloud() {
    points_.reserve(points.size());
    points_.insert(points_.end(), points.begin(), points.end());
}

struct FieldReader {
    template<typename... T>
    using It_ = std::variant<sensor_msgs::PointCloud2ConstIterator<T>...>;
    using It = It_<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, float, double>;
    It it_;

    template<typename T>
    struct Type;
    template<typename T>
    struct Type<sensor_msgs::PointCloud2ConstIterator<T>> { using type = T; };

    FieldReader(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg, uint8_t datatype, const std::string& name) : 
        it_([&]() -> FieldReader::It {
        using namespace sensor_msgs;
        using namespace sensor_msgs::msg;
        switch(datatype) {
        case PointField::INT8:    return It { PointCloud2ConstIterator<i1>(*msg, name) };
        case PointField::UINT8:   return It { PointCloud2ConstIterator<u1>(*msg, name) };
        case PointField::INT16:   return It { PointCloud2ConstIterator<i2>(*msg, name) };
        case PointField::UINT16:  return It { PointCloud2ConstIterator<u2>(*msg, name) };
        case PointField::INT32:   return It { PointCloud2ConstIterator<i4>(*msg, name) };
        case PointField::UINT32:  return It { PointCloud2ConstIterator<u4>(*msg, name) };
        case PointField::FLOAT32: return It { PointCloud2ConstIterator<f4>(*msg, name) };
        case PointField::FLOAT64: return It { PointCloud2ConstIterator<f8>(*msg, name) };
        default: throw std::logic_error("unsupported field type");
        }
    }()) { /* STUB */ }

    void read(Cloud::Field& field) noexcept {
        std::visit([&field](auto&& it) {
            using T = typename FieldReader::Type<std::decay_t<decltype(it)>>::type;
            std::get<std::vector<T>>(field.data_).emplace_back(*it);
            ++it;
        }, it_);
    }
};

Cloud::Cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) noexcept {
    assert(msg->fields.size() >= 3);
    assert(msg->fields[0].name == "x");
    assert(msg->fields[1].name == "y");
    assert(msg->fields[2].name == "z");

    sensor_msgs::PointCloud2ConstIterator<float> msg_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> msg_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> msg_z(*msg, "z");

    auto it = msg->fields.cbegin();
    static std::set<std::string> XYZ = {"x", "y", "z"};
    while(XYZ.count(it->name) > 0) ++it;

    std::vector<FieldReader> msg_it;
    std::for_each(it, msg->fields.cend(), [&](const auto& field) {
        desc_.add_field(field.datatype, field.name);
        fields_.emplace_back(field.datatype);
        msg_it.emplace_back(msg, field.datatype, field.name);
    });

    size_t point_count = msg->height * msg->width;
    points_.reserve(point_count);

    size_t field_count = msg_it.size();
    assert(field_count == desc_.attrs_.size());
    assert(field_count == desc_.names_.size());
    for(auto& field : fields_) {
        std::visit([point_count](auto& it) { it.reserve(point_count); }, field.data_);
    }

    for(size_t i = 0; i < point_count; ++i, ++msg_x, ++msg_y, ++msg_z) {
        points_.emplace_back(*msg_x, *msg_y, *msg_z);
        for(size_t j = 0; j < field_count; ++j) msg_it[j].read(fields_[j]);
    }
}

Cloud::Cloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) noexcept :
        Cloud(std::const_pointer_cast<const sensor_msgs::msg::PointCloud2>(msg)) { /* STUB */ }

Cloud::Cloud(Cloud&& other) noexcept : desc_(other.desc_), 
    points_(std::move(other.points_)), fields_(std::move(other.fields_)) {
    other.points_.clear();
    other.fields_.clear();
    for(size_t i = 0; i < fields_.size(); ++i) {
        other.fields_.emplace_back(other.desc_.attrs_[i].first);
    }
}

Cloud& Cloud::operator=(Cloud&& other) noexcept {
    if(this == &other) return *this;
    desc_ = other.desc_;
    points_ = std::move(other.points_);
    fields_ = std::move(other.fields_);
    other.points_.clear();
    other.fields_.clear();
    for(size_t i = 0; i < fields_.size(); ++i) {
        other.fields_.emplace_back(other.desc_.attrs_[i].first);
    }
    return *this;
}

void Cloud::clear() noexcept {
    points_.clear();
    for(auto& field : fields_) field.clear();
}

void Cloud::reserve(size_t new_cap) {
    points_.reserve(new_cap);
    for(auto& field : fields_) field.reserve(new_cap);
}

void Cloud::resize(size_t count) {
    points_.resize(count);
    for(Field& field : fields_) field.resize(count);
}

std::span<const Eigen::Vector3f> Cloud::points() const noexcept { 
    return points_; 
}

std::span<Eigen::Vector3f> Cloud::points() noexcept { 
    return points_; 
}

template<typename T>
std::span<const T> Cloud::field(const std::string& name) const {
    try {
        return std::get<std::vector<T>>(fields_[desc_.names_.at(name)].data_);
    } catch(const std::out_of_range& e) {
        throw std::runtime_error("field not found in cloud");
    }
}

template std::span<const i1> Cloud::field(const std::string&) const;
template std::span<const u1> Cloud::field(const std::string&) const;
template std::span<const i2> Cloud::field(const std::string&) const;
template std::span<const u2> Cloud::field(const std::string&) const;
template std::span<const i4> Cloud::field(const std::string&) const;
template std::span<const u4> Cloud::field(const std::string&) const;
template std::span<const f4> Cloud::field(const std::string&) const;
template std::span<const f8> Cloud::field(const std::string&) const;

template<typename T>
std::span<T> Cloud::field(const std::string& name) {
    try {
        return std::get<std::vector<T>>(fields_[desc_.names_.at(name)].data_);
    } catch(const std::out_of_range& e) {
        throw std::runtime_error("field not found in cloud");
    }
}

template std::span<i1> Cloud::field(const std::string&);
template std::span<u1> Cloud::field(const std::string&);
template std::span<i2> Cloud::field(const std::string&);
template std::span<u2> Cloud::field(const std::string&);
template std::span<i4> Cloud::field(const std::string&);
template std::span<u4> Cloud::field(const std::string&);
template std::span<f4> Cloud::field(const std::string&);
template std::span<f8> Cloud::field(const std::string&);

#ifdef GENERATE_CLOUD_PYBIND_MODULE
py::object Cloud::points_as_py_object() const noexcept {
    return py::cast(points_, py::return_value_policy::reference);
}

template<typename T>
py::object Cloud::field_as_py_object(const std::string& name) const {
    try {
        return py::cast(std::get<std::vector<T>>(fields_[desc_.names_.at(name)].data_), py::return_value_policy::reference);
    } catch(const std::out_of_range& e) {
        throw std::runtime_error("field not found in cloud");
    }
}

template py::object Cloud::field_as_py_object<i1>(const std::string&) const;
template py::object Cloud::field_as_py_object<u1>(const std::string&) const;
template py::object Cloud::field_as_py_object<i2>(const std::string&) const;
template py::object Cloud::field_as_py_object<u2>(const std::string&) const;
template py::object Cloud::field_as_py_object<i4>(const std::string&) const;
template py::object Cloud::field_as_py_object<u4>(const std::string&) const;
template py::object Cloud::field_as_py_object<f4>(const std::string&) const;
template py::object Cloud::field_as_py_object<f8>(const std::string&) const;
#endif // GENERATE_CLOUD_PYBIND_MODULE

void Cloud::downsample(float voxel_size) {
    // TODO: Could this be static to prevent allocation every frame??
    tsl::robin_map<Eigen::Vector3i, size_t> grid;
    grid.reserve(Cloud::size());
    size_t i_write = 0;
    for(size_t i_read = 0; i_read < Cloud::size(); ++i_read) {
        const Eigen::Vector3i voxel = point_to_voxel(points_[i_read], voxel_size);
        if(!grid.contains(voxel)) {
            grid.insert({voxel, i_write});
            points_[i_write] = points_[i_read];
            for(size_t i = 0; i < fields_.size(); ++i) {
                // fields_[i].copy(fields_[i], other.index, index);
                std::visit([i_read, i_write](auto& vec) {
                    vec[i_write] = vec[i_read];
                }, fields_[i].data_);
            }
            ++i_write;
        }
    }
    Cloud::resize(i_write);
}

void Cloud::filter_by_distance(const Eigen::Vector3f& origin, float minima, float maxima) {
    float minima_sq = minima * minima;
    float maxima_sq = maxima * maxima;
    const size_t n = points_.size();
    size_t i_write = 0;
    for(size_t i_read = 0; i_read < n; ++i_read) {
        const float d_sq = (points_[i_read] - origin).squaredNorm();
        if(d_sq < minima_sq || d_sq > maxima_sq) continue;
        if(i_write != i_read) {
            points_[i_write] = points_[i_read];
            for(Field& f : fields_) {
                std::visit([i_write, i_read](auto& vec) { vec[i_write] = vec[i_read]; }, f.data_);
            }
        }
        ++i_write;
    }
    points_.resize(i_write);
    for(Field& f : fields_) {
        std::visit([i_write](auto& vec) { vec.resize(i_write); }, f.data_);
    }
}

struct FieldWriter {
    template<typename... T>
    using It_ = std::variant<sensor_msgs::PointCloud2Iterator<T>...>;
    using It = It_<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, float, double>;
    It it_;

    template<typename T>
    struct Type;
    template<typename T>
    struct Type<sensor_msgs::PointCloud2Iterator<T>> { using type = T; };

    FieldWriter(sensor_msgs::msg::PointCloud2& msg, uint8_t datatype, const std::string& name) : 
        it_([&]() -> FieldWriter::It {
        using namespace sensor_msgs;
        using namespace sensor_msgs::msg;
        switch(datatype) {
        case PointField::INT8:    return It { PointCloud2Iterator<i1>(msg, name) };
        case PointField::UINT8:   return It { PointCloud2Iterator<u1>(msg, name) };
        case PointField::INT16:   return It { PointCloud2Iterator<i2>(msg, name) };
        case PointField::UINT16:  return It { PointCloud2Iterator<u2>(msg, name) };
        case PointField::INT32:   return It { PointCloud2Iterator<i4>(msg, name) };
        case PointField::UINT32:  return It { PointCloud2Iterator<u4>(msg, name) };
        case PointField::FLOAT32: return It { PointCloud2Iterator<f4>(msg, name) };
        case PointField::FLOAT64: return It { PointCloud2Iterator<f8>(msg, name) };
        default: throw std::logic_error("unsupported field type");
        }
    }()) { /* STUB */ }

    template<typename T>
    void write(T scalar) noexcept {
        std::visit([scalar](auto&& it) {
            *it = scalar;
            ++it;
        }, it_);
    }
};

sensor_msgs::msg::PointCloud2 Cloud::message(const std_msgs::msg::Header& header) const noexcept {
    sensor_msgs::msg::PointCloud2 msg;
    sensor_msgs::PointCloud2Modifier modifier(msg);
    msg.header = header;
    msg.header.frame_id = fix_frame_id(msg.header.frame_id);
    msg.fields.clear();
    // calculate offset given fields
    int offset = 0;
    offset = addPointField(msg, "x", 1, sensor_msgs::msg::PointField::FLOAT32, offset);
    offset = addPointField(msg, "y", 1, sensor_msgs::msg::PointField::FLOAT32, offset);
    offset = addPointField(msg, "z", 1, sensor_msgs::msg::PointField::FLOAT32, offset);
    // generic fields
    for(const auto& [datatype, name] : desc_.attrs_) {
        offset = addPointField(msg, name, 1, datatype, offset);
    }
    msg.point_step = offset;
    msg.row_step = msg.width * msg.point_step;
    msg.data.resize(static_cast<size_t>(msg.height) * msg.row_step);
    modifier.resize(Cloud::size());
    // position iterators
    sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");
    // build iterators for each field
    std::vector<FieldWriter> msg_it;
    for(const auto& [datatype, name] : desc_.attrs_) {
        msg_it.emplace_back(msg, datatype, name);
    }
    size_t field_count = msg_it.size();
    assert(field_count == desc_.attrs_.size());
    assert(field_count == desc_.names_.size());
    // populate the message
    for(size_t i = 0; i < Cloud::size(); ++i, ++msg_x, ++msg_y, ++msg_z) {
        const Eigen::Vector3f& point = points_[i];
        *msg_x = static_cast<float>(point.x());
        *msg_y = static_cast<float>(point.y());
        *msg_z = static_cast<float>(point.z());
        for(size_t j = 0; j < field_count; ++j) {
            std::visit([&msg_it, j, i](auto& vec) {
                msg_it[j].write(vec[i]);
            }, fields_[j].data_);
        }
    }
    return msg;
}

Cloud::Field::Field(uint8_t datatype) : data_([&]() -> Cloud::Field::Data {
    using namespace sensor_msgs::msg;
    switch(datatype) {
    case PointField::INT8:    return Data { std::vector<i1>(0) };
    case PointField::UINT8:   return Data { std::vector<u1>(0) };
    case PointField::INT16:   return Data { std::vector<i2>(0) };
    case PointField::UINT16:  return Data { std::vector<u2>(0) };
    case PointField::INT32:   return Data { std::vector<i4>(0) };
    case PointField::UINT32:  return Data { std::vector<u4>(0) };
    case PointField::FLOAT32: return Data { std::vector<f4>(0) };
    case PointField::FLOAT64: return Data { std::vector<f8>(0) };
    default: throw std::logic_error("unsupported field type");
    }
}()) { /* STUB */ }

void Cloud::Field::clear() noexcept {
    std::visit([](auto& vec) { vec.clear(); }, data_);
}

void Cloud::Field::reserve(size_t new_cap) {
    std::visit([new_cap](auto& vec) { vec.reserve(new_cap); }, data_);
}

void Cloud::Field::resize(size_t count) {
    std::visit([count](auto& vec) { vec.resize(count); }, data_);
}

template<typename T>
T& Cloud::Field::operator[](size_t index) {
    return std::get<std::vector<T>>(data_)[index];
}

template i1& Cloud::Field::operator[](size_t index);
template u1& Cloud::Field::operator[](size_t index);
template i2& Cloud::Field::operator[](size_t index);
template u2& Cloud::Field::operator[](size_t index);
template i4& Cloud::Field::operator[](size_t index);
template u4& Cloud::Field::operator[](size_t index);
template f4& Cloud::Field::operator[](size_t index);
template f8& Cloud::Field::operator[](size_t index);

template<typename T>
const T& Cloud::Field::operator[](size_t index) const {
    return std::get<std::vector<T>>(data_)[index];
}

template const i1& Cloud::Field::operator[](size_t index) const;
template const u1& Cloud::Field::operator[](size_t index) const;
template const i2& Cloud::Field::operator[](size_t index) const;
template const u2& Cloud::Field::operator[](size_t index) const;
template const i4& Cloud::Field::operator[](size_t index) const;
template const u4& Cloud::Field::operator[](size_t index) const;
template const f4& Cloud::Field::operator[](size_t index) const;
template const f8& Cloud::Field::operator[](size_t index) const;

template<typename T>
void Cloud::add_field(const std::string& name, const std::vector<T>& scalars) noexcept {
    assert(Cloud::size() == scalars.size());
    assert(!desc_.names_.contains(name));
    uint8_t datatype;
    if      constexpr (std::is_same_v<T, i1>) datatype = sensor_msgs::msg::PointField::INT8;
    else if constexpr (std::is_same_v<T, u1>) datatype = sensor_msgs::msg::PointField::UINT8;
    else if constexpr (std::is_same_v<T, i2>) datatype = sensor_msgs::msg::PointField::INT16;
    else if constexpr (std::is_same_v<T, u2>) datatype = sensor_msgs::msg::PointField::UINT16;
    else if constexpr (std::is_same_v<T, i4>) datatype = sensor_msgs::msg::PointField::INT32;
    else if constexpr (std::is_same_v<T, u4>) datatype = sensor_msgs::msg::PointField::UINT32;
    else if constexpr (std::is_same_v<T, f4>) datatype = sensor_msgs::msg::PointField::FLOAT32;
    else if constexpr (std::is_same_v<T, f8>) datatype = sensor_msgs::msg::PointField::FLOAT64;
    else static_assert(Cloud::False<T>, "unsupported field type");
    desc_.add_field(datatype, name);
    fields_.emplace_back(scalars);
}

template void Cloud::add_field<i1>(const std::string&, const std::vector<i1>&) noexcept;
template void Cloud::add_field<u1>(const std::string&, const std::vector<u1>&) noexcept;
template void Cloud::add_field<i2>(const std::string&, const std::vector<i2>&) noexcept;
template void Cloud::add_field<u2>(const std::string&, const std::vector<u2>&) noexcept;
template void Cloud::add_field<i4>(const std::string&, const std::vector<i4>&) noexcept;
template void Cloud::add_field<u4>(const std::string&, const std::vector<u4>&) noexcept;
template void Cloud::add_field<f4>(const std::string&, const std::vector<f4>&) noexcept;
template void Cloud::add_field<f8>(const std::string&, const std::vector<f8>&) noexcept;

template<typename T>
void Cloud::add_field(const std::string& name, const T& scalar) noexcept {
    std::vector<T> scalars(Cloud::size(), scalar);
    Cloud::add_field(name, scalars);
}

template void Cloud::add_field<i1>(const std::string&, const i1&) noexcept;
template void Cloud::add_field<u1>(const std::string&, const u1&) noexcept;
template void Cloud::add_field<i2>(const std::string&, const i2&) noexcept;
template void Cloud::add_field<u2>(const std::string&, const u2&) noexcept;
template void Cloud::add_field<i4>(const std::string&, const i4&) noexcept;
template void Cloud::add_field<u4>(const std::string&, const u4&) noexcept;
template void Cloud::add_field<f4>(const std::string&, const f4&) noexcept;
template void Cloud::add_field<f8>(const std::string&, const f8&) noexcept;

#ifdef GENERATE_CLOUD_PYBIND_MODULE
template<typename T>
void Cloud::add_field_from_py_object(const std::string& name, py::object obj) noexcept {
    Cloud::add_field(name, obj.cast<std::vector<T>>());
}

template void Cloud::add_field_from_py_object<i1>(const std::string&, py::object) noexcept;
template void Cloud::add_field_from_py_object<u1>(const std::string&, py::object) noexcept;
template void Cloud::add_field_from_py_object<i2>(const std::string&, py::object) noexcept;
template void Cloud::add_field_from_py_object<u2>(const std::string&, py::object) noexcept;
template void Cloud::add_field_from_py_object<i4>(const std::string&, py::object) noexcept;
template void Cloud::add_field_from_py_object<u4>(const std::string&, py::object) noexcept;
template void Cloud::add_field_from_py_object<f4>(const std::string&, py::object) noexcept;
template void Cloud::add_field_from_py_object<f8>(const std::string&, py::object) noexcept;
#endif // GENERATE_CLOUD_PYBIND_MODULE
} // namespace util

#ifdef GENERATE_CLOUD_PYBIND_MODULE
#include "stl_vector_eigen.h"

// binding code here still works because it just needs to be visible in the final binary
PYBIND11_EMBEDDED_MODULE(cloud_bindings, m) {
    // TODO: ENSURE THIS DOESN'T COPY!
    pybind11::bind_vector<FieldI1>(m, "FieldI1");
    pybind11::bind_vector<FieldU1>(m, "FieldU1");
    pybind11::bind_vector<FieldI2>(m, "FieldI2");
    pybind11::bind_vector<FieldU2>(m, "FieldU2");
    pybind11::bind_vector<FieldI4>(m, "FieldI4");
    pybind11::bind_vector<FieldU4>(m, "FieldU4");
    pybind11::bind_vector<FieldF4>(m, "FieldF4");
    pybind11::bind_vector<FieldF8>(m, "FieldF8");
    // (n, 3) numpy array
    pybind_eigen_vector_of_vector<Eigen::Vector3f>(m, "Points", "Points", pybind11::py_array_to_vectors<Eigen::Vector3f>);
}
#endif // GENERATE_CLOUD_PYBIND_MODULE