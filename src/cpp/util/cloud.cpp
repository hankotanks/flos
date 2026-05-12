#include "cloud.h"

// system
#include <set>
#include <regex>

namespace {
inline std::string fix_frame_id(const std::string &frame_id) {
    return std::regex_replace(frame_id, std::regex("^/"), "");
}
} // private scope

namespace util {
Cloud::Cloud(const Descriptor& desc) noexcept : desc_(desc) {
    assert(desc_.attrs_.size() == desc_.names_.size());
    for(size_t i = 0; i < desc_.attrs_.size(); ++i) {
        fields_.emplace_back(desc_.attrs_[i].first);
    }
}

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

    std::vector<Cloud::FieldReader> msg_it;
    std::for_each(it, msg->fields.cend(), [&](const auto& field) {
        desc_.add_field(field.datatype, field.name);
        fields_.emplace_back(field.datatype);
        msg_it.emplace_back(msg, field.datatype, field.name);
    });

    size_t point_count = msg->height * msg->width;
    points_.reserve(point_count);

    size_t field_count = msg_it.size();
    assert(field_count == desc_.size());
    for(auto& field : fields_) {
        std::visit([point_count](auto& it) { it.reserve(point_count); }, field.data_);
    }

    for(size_t i = 0; i < point_count; ++i, ++msg_x, ++msg_y, ++msg_z) {
        points_.emplace_back(*msg_x, *msg_y, *msg_z);
        for(size_t j = 0; j < field_count; ++j) msg_it[j].read(fields_[j]);
    }
}

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
    std::vector<Cloud::FieldWriter> msg_it;
    for(const auto& [datatype, name] : desc_.attrs_) {
        msg_it.emplace_back(msg, datatype, name);
    }
    size_t field_count = msg_it.size();
    assert(field_count == cloud_desc_.attrs.size());
    assert(field_count == cloud_desc_.names.size());
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

void Cloud::add_point(const Cloud& other, size_t i) noexcept {
    assert(desc_ == other.desc_);
    assert(fields_.size() == other.fields_size());
    points_.emplace_back(other.points_[i]);
    for(size_t j = 0; j < fields_.size(); ++j) {
        std::visit([&](auto& vec) {
            using T = typename std::decay_t<decltype(vec)>;
            vec.emplace_back(std::get<T>(other.fields_[j].data_)[i]);
        }, fields_[j].data_);
    }
}

Cloud::Point Cloud::add_point(const Cloud::Point& point) noexcept {
    size_t index = Cloud::size();
    Cloud::add_point(*point.owner_, point.index);
    return {this, index};
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

Cloud::FieldReader::FieldReader(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg, uint8_t datatype, const std::string& name) : 
    it_([&]() -> Cloud::FieldReader::It {
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

void Cloud::FieldReader::read(Cloud::Field& field) noexcept {
    std::visit([&field](auto&& it) {
        using T = typename Cloud::FieldReader::Type<std::decay_t<decltype(it)>>::type;
        std::get<std::vector<T>>(field.data_).emplace_back(*it);
        ++it;
    }, it_);
}

Cloud::FieldWriter::FieldWriter(sensor_msgs::msg::PointCloud2& msg, uint8_t datatype, const std::string& name) : 
    it_([&]() -> Cloud::FieldWriter::It {
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
}

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