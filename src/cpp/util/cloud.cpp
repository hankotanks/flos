#include "cloud.h"

// system
#include <set>
#include <regex>
#include <stdexcept>

namespace {
inline std::string fix_frame_id(const std::string &frame_id) {
    return std::regex_replace(frame_id, std::regex("^/"), "");
}
} // private scope

namespace util {
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

Cloud::Cloud(const std::shared_ptr<const Descriptor>& desc, bool shared) noexcept : shared_(shared),
    desc_(shared ? std::shared_ptr<Descriptor>(desc, const_cast<Descriptor*>(desc.get())) : std::make_shared<Descriptor>(*desc)) {
    assert(desc_->attrs_.size() == desc_->names_.size());
    for(size_t i = 0; i < desc_->attrs_.size(); ++i) {
        fields_.emplace_back(desc_->attrs_[i].first);
    }
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
        case PointField::INT8:    return It { PointCloud2ConstIterator<int8_t>(*msg, name) };
        case PointField::UINT8:   return It { PointCloud2ConstIterator<uint8_t>(*msg, name) };
        case PointField::INT16:   return It { PointCloud2ConstIterator<int16_t>(*msg, name) };
        case PointField::UINT16:  return It { PointCloud2ConstIterator<uint16_t>(*msg, name) };
        case PointField::INT32:   return It { PointCloud2ConstIterator<int32_t>(*msg, name) };
        case PointField::UINT32:  return It { PointCloud2ConstIterator<uint32_t>(*msg, name) };
        case PointField::FLOAT32: return It { PointCloud2ConstIterator<float>(*msg, name) };
        case PointField::FLOAT64: return It { PointCloud2ConstIterator<double>(*msg, name) };
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

    desc_ = std::make_shared<Cloud::Descriptor>();

    sensor_msgs::PointCloud2ConstIterator<float> msg_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> msg_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> msg_z(*msg, "z");

    auto it = msg->fields.cbegin();
    static std::set<std::string> XYZ = {"x", "y", "z"};
    while(XYZ.count(it->name) > 0) ++it;

    std::vector<FieldReader> msg_it;
    std::for_each(it, msg->fields.cend(), [&](const auto& field) {
        desc_->add_field(field.datatype, field.name);
        fields_.emplace_back(field.datatype);
        msg_it.emplace_back(msg, field.datatype, field.name);
    });

    size_t point_count = msg->height * msg->width;
    points_.reserve(point_count);

    size_t field_count = msg_it.size();
    assert(field_count == desc_->size());
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

Cloud::Cloud(const Cloud& other) noexcept : desc_(std::make_shared<Cloud::Descriptor>(*(other.desc_))), 
    points_(other.points_), fields_(other.fields_) { /* STUB */ }

Cloud::Cloud(Cloud&& other) noexcept : desc_(std::make_shared<Cloud::Descriptor>(*(other.desc_))), 
    points_(std::move(other.points_)), fields_(std::move(other.fields_)) {
    other.points_.clear();
    other.fields_.clear();
    for(size_t i = 0; i < fields_.size(); ++i) {
        other.fields_.emplace_back(other.desc_->attrs_[i].first);
    }
}

Cloud& Cloud::operator=(Cloud&& other) noexcept {
    if(this == &other) return *this;
    desc_ = std::make_shared<Cloud::Descriptor>(*(other.desc_));
    points_ = std::move(other.points_);
    fields_ = std::move(other.fields_);
    other.points_.clear();
    other.fields_.clear();
    for(size_t i = 0; i < fields_.size(); ++i) {
        other.fields_.emplace_back(other.desc_->attrs_[i].first);
    }
    return *this;
}

size_t Cloud::size() const noexcept { 
    assert(desc_->attrs_.size() == desc_->names_.size());
    assert(desc_->attrs_.size() == points_.size());
    return points_.size(); 
}

bool Cloud::empty() const noexcept { 
    return Cloud::size() == 0;
}

void Cloud::clear() noexcept {
    points_.clear();
    for(auto& field : fields_) field.clear();
}

void Cloud::reserve(size_t n) noexcept {
    points_.reserve(n);
    for(auto& field : fields_) field.reserve(n);
}

std::shared_ptr<const Cloud::Descriptor> Cloud::descriptor() const noexcept {
    return desc_;
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
        return std::get<std::vector<T>>(fields_[desc_->names_.at(name)].data_);
    } catch(const std::out_of_range& e) {
        throw std::runtime_error("field not found in cloud");
    }
}

template std::span<const int8_t>  Cloud::field(const std::string&) const;
template std::span<const uint8_t> Cloud::field(const std::string&) const;
template std::span<const int16_t> Cloud::field(const std::string&) const;
template std::span<const uint16_t> Cloud::field(const std::string&) const;
template std::span<const int32_t> Cloud::field(const std::string&) const;
template std::span<const uint32_t> Cloud::field(const std::string&) const;
template std::span<const float> Cloud::field(const std::string&) const;
template std::span<const double> Cloud::field(const std::string&) const;

template<typename T>
std::span<T> Cloud::field(const std::string& name) {
    try {
        return std::get<std::vector<T>>(fields_[desc_->names_.at(name)].data_);
    } catch(const std::out_of_range& e) {
        throw std::runtime_error("field not found in cloud");
    }
}

template std::span<int8_t>  Cloud::field(const std::string&);
template std::span<uint8_t> Cloud::field(const std::string&);
template std::span<int16_t> Cloud::field(const std::string&);
template std::span<uint16_t> Cloud::field(const std::string&);
template std::span<int32_t> Cloud::field(const std::string&);
template std::span<uint32_t> Cloud::field(const std::string&);
template std::span<float> Cloud::field(const std::string&);
template std::span<double> Cloud::field(const std::string&);

Cloud::Point Cloud::add_point(const Cloud::ConstPoint& point) noexcept {
    assert(desc_ == point.owner_->desc_);
    assert(fields_.size() == point.owner_->fields_.size());
    size_t index = Cloud::size();    
    points_.emplace_back(point.owner_->points_[point.index_]);
    for(size_t j = 0; j < fields_.size(); ++j) {
        std::visit([&](auto& vec) {
            using T = typename std::decay_t<decltype(vec)>;
            vec.emplace_back(std::get<T>(point.owner_->fields_[j].data_)[point.index_]);
        }, fields_[j].data_);
    }
    return Cloud::Point{this, index};
}

template<typename T>
void Cloud::add_field(const std::string& name, const std::vector<T>& scalars) {
    // TODO: throwing will slow this down when called in SegmentationNode::loop
    if(Cloud::size() != scalars.size()) throw std::logic_error("incorrect number of scalars");
    if(desc_->names_.contains(name)) throw std::runtime_error("field already exists");
    uint8_t datatype;
    if constexpr (std::is_same_v<T, int8_t>)        datatype = sensor_msgs::msg::PointField::INT8;
    else if constexpr (std::is_same_v<T, uint8_t>)  datatype = sensor_msgs::msg::PointField::UINT8;
    else if constexpr (std::is_same_v<T, int16_t>)  datatype = sensor_msgs::msg::PointField::INT16;
    else if constexpr (std::is_same_v<T, uint16_t>) datatype = sensor_msgs::msg::PointField::UINT16;
    else if constexpr (std::is_same_v<T, int32_t>)  datatype = sensor_msgs::msg::PointField::INT32;
    else if constexpr (std::is_same_v<T, uint32_t>) datatype = sensor_msgs::msg::PointField::UINT32;
    else if constexpr (std::is_same_v<T, float>)    datatype = sensor_msgs::msg::PointField::FLOAT32;
    else if constexpr (std::is_same_v<T, double>)   datatype = sensor_msgs::msg::PointField::FLOAT64;
    else static_assert(Cloud::False<T>, "unsupported field type");
    desc_->add_field(datatype, name);
    fields_.emplace_back(scalars);
}

template void Cloud::add_field<int8_t>(const std::string&, const std::vector<int8_t>&);
template void Cloud::add_field<uint8_t>(const std::string&, const std::vector<uint8_t>&);
template void Cloud::add_field<int16_t>(const std::string&, const std::vector<int16_t>&);
template void Cloud::add_field<uint16_t>(const std::string&, const std::vector<uint16_t>&);
template void Cloud::add_field<int32_t>(const std::string&, const std::vector<int32_t>&);
template void Cloud::add_field<uint32_t>(const std::string&, const std::vector<uint32_t>&);
template void Cloud::add_field<float>(const std::string&, const std::vector<float>&);
template void Cloud::add_field<double>(const std::string&, const std::vector<double>&);

#ifdef GENERATE_CLOUD_PYBIND_MODULE
py::object Cloud::points_as_py_object() const noexcept {
    return py::cast(points_, py::return_value_policy::reference);
}

template<typename T>
py::object Cloud::field_as_py_object(const std::string& name) const {
    try {
        return py::cast(std::get<std::vector<T>>(fields_[desc_->names_.at(name)].data_), py::return_value_policy::reference);
    } catch(const std::out_of_range& e) {
        throw std::runtime_error("field not found in cloud");
    }
}

template py::object Cloud::field_as_py_object<int8_t>(const std::string&) const;
template py::object Cloud::field_as_py_object<uint8_t>(const std::string&) const;
template py::object Cloud::field_as_py_object<int16_t>(const std::string&) const;
template py::object Cloud::field_as_py_object<uint16_t>(const std::string&) const;
template py::object Cloud::field_as_py_object<int32_t>(const std::string&) const;
template py::object Cloud::field_as_py_object<uint32_t>(const std::string&) const;
template py::object Cloud::field_as_py_object<float>(const std::string&) const;
template py::object Cloud::field_as_py_object<double>(const std::string&) const;

template<typename T>
void Cloud::add_field_from_py_object(const std::string& name, py::object obj) {
    Cloud::add_field(name, obj.cast<std::vector<T>>());
}

template void Cloud::add_field_from_py_object<int8_t>(const std::string&, py::object);
template void Cloud::add_field_from_py_object<uint8_t>(const std::string&, py::object);
template void Cloud::add_field_from_py_object<int16_t>(const std::string&, py::object);
template void Cloud::add_field_from_py_object<uint16_t>(const std::string&, py::object);
template void Cloud::add_field_from_py_object<int32_t>(const std::string&, py::object);
template void Cloud::add_field_from_py_object<uint32_t>(const std::string&, py::object);
template void Cloud::add_field_from_py_object<float>(const std::string&, py::object);
template void Cloud::add_field_from_py_object<double>(const std::string&, py::object);
#endif // GENERATE_CLOUD_PYBIND_MODULE

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
        case PointField::INT8:    return It { PointCloud2Iterator<int8_t>(msg, name) };
        case PointField::UINT8:   return It { PointCloud2Iterator<uint8_t>(msg, name) };
        case PointField::INT16:   return It { PointCloud2Iterator<int16_t>(msg, name) };
        case PointField::UINT16:  return It { PointCloud2Iterator<uint16_t>(msg, name) };
        case PointField::INT32:   return It { PointCloud2Iterator<int32_t>(msg, name) };
        case PointField::UINT32:  return It { PointCloud2Iterator<uint32_t>(msg, name) };
        case PointField::FLOAT32: return It { PointCloud2Iterator<float>(msg, name) };
        case PointField::FLOAT64: return It { PointCloud2Iterator<double>(msg, name) };
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
    for(const auto& [datatype, name] : desc_->attrs_) {
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
    for(const auto& [datatype, name] : desc_->attrs_) {
        msg_it.emplace_back(msg, datatype, name);
    }
    size_t field_count = msg_it.size();
    assert(field_count == cloud_desc_->attrs.size());
    assert(field_count == cloud_desc_->names.size());
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
    case PointField::INT8:    return Data { std::vector<int8_t>(0) };
    case PointField::UINT8:   return Data { std::vector<uint8_t>(0) };
    case PointField::INT16:   return Data { std::vector<int16_t>(0) };
    case PointField::UINT16:  return Data { std::vector<uint16_t>(0) };
    case PointField::INT32:   return Data { std::vector<int32_t>(0) };
    case PointField::UINT32:  return Data { std::vector<uint32_t>(0) };
    case PointField::FLOAT32: return Data { std::vector<float>(0) };
    case PointField::FLOAT64: return Data { std::vector<double>(0) };
    default: throw std::logic_error("unsupported field type");
    }
}()) { /* STUB */ }

void Cloud::Field::clear() noexcept {
    std::visit([](auto& vec) { vec.clear(); }, data_);
}

void Cloud::Field::reserve(size_t n) noexcept {
    std::visit([n](auto& vec) { vec.reserve(n); }, data_);
}

} // namespace util

#ifdef GENERATE_CLOUD_PYBIND_MODULE
#include "stl_vector_eigen.h"

// binding code here still works because it just needs to be visible in the final binary
PYBIND11_EMBEDDED_MODULE(cloud_bindings, m) {
    // TODO: ENSURE THIS DOESN'T COPY!
    pybind11::bind_vector<FieldInt8>(m, "FieldInt8");
    pybind11::bind_vector<FieldUInt8>(m, "FieldUInt8");
    pybind11::bind_vector<FieldInt16>(m, "FieldInt16");
    pybind11::bind_vector<FieldUInt16>(m, "FieldUInt16");
    pybind11::bind_vector<FieldInt32>(m, "FieldInt32");
    pybind11::bind_vector<FieldUInt32>(m, "FieldUInt32");
    pybind11::bind_vector<FieldFloat32>(m, "FieldFloat32");
    pybind11::bind_vector<FieldFloat64>(m, "FieldFloat64");
    // (n, 3) numpy array
    pybind_eigen_vector_of_vector<Eigen::Vector3f>(m, "Points", "Points", pybind11::py_array_to_vectors<Eigen::Vector3f>);
}
#endif // GENERATE_CLOUD_PYBIND_MODULE