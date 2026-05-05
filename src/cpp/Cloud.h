#pragma once

#include <map>
#include <set>
#include <stdexcept>
#include <vector>
#include <regex>

#include <Eigen/Dense>

#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace {
    inline std::string fix_frame_id(const std::string &frame_id) {
        return std::regex_replace(frame_id, std::regex("^/"), "");
    }
} // private scope

namespace cloud {

struct CloudDescriptor {
    friend class Cloud;
private:
    CloudDescriptor() = default;
    CloudDescriptor(const CloudDescriptor&) = default;
    CloudDescriptor(CloudDescriptor&&) = delete;
    CloudDescriptor& operator=(const CloudDescriptor&) = default;
    CloudDescriptor& operator=(CloudDescriptor&&) = delete;
    
private:
    bool operator==(const CloudDescriptor& other) {
        return attrs == other.attrs && names == other.names;
    }

private:
    std::deque<std::pair<uint8_t, std::string>> attrs;
    std::map<std::string_view, size_t> names;
};

class Cloud {
public:
    Cloud(const CloudDescriptor& desc) : cloud_desc_(desc) { /* STUB */ }

    Cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
        assert(msg->fields.size() >= 3);
        assert(msg->fields[0].name == "x");
        assert(msg->fields[0].name == "y");
        assert(msg->fields[0].name == "z");

        sensor_msgs::PointCloud2ConstIterator<float> msg_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> msg_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> msg_z(*msg, "z");

        auto it = msg->fields.cbegin();
        static std::set<std::string> XYZ = {"x", "y", "z"};
        while(XYZ.count(it->name) > 0) ++it;

        std::vector<ScalarConstIterator> msg_it;
        std::for_each(it, msg->fields.cend(), [&](const auto& field) {
            cloud_desc_.attrs.emplace_back(field.datatype, field.name);
            const std::string& name = cloud_desc_.attrs.back().second;
            cloud_desc_.names.emplace(std::string_view(name), cloud_desc_.attrs.size() - 1);
            msg_it.emplace_back(msg, field.datatype, field.name);
        });

        size_t field_count = msg_it.size();
        assert(field_count == cloud_desc_.attrs.size());
        assert(field_count == cloud_desc_.names.size());
        
        size_t point_count = msg->height * msg->width;
        points_.reserve(point_count);
        field_scalars_.reserve(point_count);
        for(size_t i = 0; i < point_count; ++i, ++msg_x, ++msg_y, ++msg_z) {
            points_.emplace_back(*msg_x, *msg_y, *msg_z);

            field_scalars_.emplace_back(field_count);

            auto& scalars = field_scalars_.back();
            for(size_t j = 0; j < scalars.size(); ++j) {
                scalars[j] = Scalar(msg_it[j], cloud_desc_.attrs[j].first);
            }
        }
    }

    Cloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) :
        Cloud(std::const_pointer_cast<const sensor_msgs::msg::PointCloud2>(msg)) { /* STUB */ }

    // descriptor is copied so that the old cloud can be reused
    Cloud(Cloud&& other) : cloud_desc_(other.cloud_desc_), 
        points_(std::move(other.points_)), 
        field_scalars_(std::move(other.field_scalars_)) { /* STUB */ }

    // same here
    Cloud& operator=(Cloud&& other) {
        if(this != &other) {
            cloud_desc_ = other.cloud_desc_;
            points_ = std::move(other.points_);
            field_scalars_ = std::move(other.field_scalars_);
        }
        return *this;
    }

    sensor_msgs::msg::PointCloud2 message(const std_msgs::msg::Header& header) noexcept {
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
        for(const auto& [datatype, name] : cloud_desc_.attrs) {
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
        std::vector<ScalarIterator> msg_it;
        for(const auto& [datatype, name] : cloud_desc_.attrs) {
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
                field_scalars_[i][j].write(msg_it[j], cloud_desc_.attrs[j].first);
            }
        }
        return msg;
    }

    size_t size() const noexcept {
        assert(points_.size() == field_scalars_.size());
        return points_.size();
    }

    bool empty() const noexcept {
        return Cloud::size() == 0;
    }

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
        else assert(false && "unsupported field type!");  
        
        cloud_desc_.attrs.emplace_back(datatype, name);
        const std::string& name_ref = cloud_desc_.attrs.back().second;
        cloud_desc_.names.emplace(std::string_view(name_ref), cloud_desc_.attrs.size() - 1);

        for(size_t i = 0; i < Cloud::size(); ++i) {
            field_scalars_[i].emplace_back(scalars[i]);
        }
    }

    void add_point(const Cloud& other, size_t i) {
        assert(cloud_desc_ == other.cloud_desc_);
        points_.emplace_back(other.points_[i]);
        field_scalars_.emplace_back(other.field_scalars_[i]);
    }

    const std::vector<Eigen::Vector3f>& get_points() const noexcept {
        return points_;
    }

    // TODO: better error handling
    // TODO: Fields should be stored as independent vectors
    template<typename T>
    std::vector<T> get_field(const std::string& name) const {
        std::vector<T> scalars(Cloud::size());
        size_t field_index;
        try {
            field_index = cloud_desc_.names.at(name);
        } catch(const std::out_of_range& e) {
            throw std::runtime_error("field not found in cloud");
        }

        for(size_t i = 0; i < Cloud::size(); ++i) {
            scalars[i] = field_scalars_[i][field_index].get<T>();
        }
        
        return scalars;
    }

private:
    Cloud() = delete;
    Cloud(const Cloud&) = delete;
    Cloud& operator=(const Cloud&) = delete;

private:
    std::vector<Eigen::Vector3f> points_;

    union ScalarIterator {
        sensor_msgs::PointCloud2Iterator<int8_t> it_int8;
        sensor_msgs::PointCloud2Iterator<uint8_t> it_uint8;
        sensor_msgs::PointCloud2Iterator<int16_t> it_int16;
        sensor_msgs::PointCloud2Iterator<uint16_t> it_uint16;
        sensor_msgs::PointCloud2Iterator<int32_t> it_int32;
        sensor_msgs::PointCloud2Iterator<uint32_t> it_uint32;
        sensor_msgs::PointCloud2Iterator<float> it_float;
        sensor_msgs::PointCloud2Iterator<double> it_double;

        ScalarIterator(sensor_msgs::msg::PointCloud2& msg, uint8_t datatype, const std::string& name) noexcept {
            switch(datatype) {
            case sensor_msgs::msg::PointField::INT8:
                it_int8 = sensor_msgs::PointCloud2Iterator<int8_t>(msg, name);
                break;
            case sensor_msgs::msg::PointField::UINT8:
                it_uint8 = sensor_msgs::PointCloud2Iterator<uint8_t>(msg, name);
                break;
            case sensor_msgs::msg::PointField::INT16:
                it_int16 = sensor_msgs::PointCloud2Iterator<int16_t>(msg, name);
                break;
            case sensor_msgs::msg::PointField::UINT16:
                it_uint16 = sensor_msgs::PointCloud2Iterator<uint16_t>(msg, name);
                break;
            case sensor_msgs::msg::PointField::INT32:
                it_int32 = sensor_msgs::PointCloud2Iterator<int32_t>(msg, name);
                break;
            case sensor_msgs::msg::PointField::UINT32:
                it_uint32 = sensor_msgs::PointCloud2Iterator<uint32_t>(msg, name);
                break;
            case sensor_msgs::msg::PointField::FLOAT32:
                it_float = sensor_msgs::PointCloud2Iterator<float>(msg, name);
                break;
            case sensor_msgs::msg::PointField::FLOAT64:
                it_double = sensor_msgs::PointCloud2Iterator<double>(msg, name);
                break;
            default: assert(false && "unsupported field type");  
            };
        }
    };

    union ScalarConstIterator {
        sensor_msgs::PointCloud2ConstIterator<int8_t> it_int8;
        sensor_msgs::PointCloud2ConstIterator<uint8_t> it_uint8;
        sensor_msgs::PointCloud2ConstIterator<int16_t> it_int16;
        sensor_msgs::PointCloud2ConstIterator<uint16_t> it_uint16;
        sensor_msgs::PointCloud2ConstIterator<int32_t> it_int32;
        sensor_msgs::PointCloud2ConstIterator<uint32_t> it_uint32;
        sensor_msgs::PointCloud2ConstIterator<float> it_float;
        sensor_msgs::PointCloud2ConstIterator<double> it_double;

        ScalarConstIterator(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg, uint8_t datatype, const std::string& name) noexcept {
            switch(datatype) {
            case sensor_msgs::msg::PointField::INT8:
                it_int8 = sensor_msgs::PointCloud2ConstIterator<int8_t>(*msg, name);
                break;
            case sensor_msgs::msg::PointField::UINT8:
                it_uint8 = sensor_msgs::PointCloud2ConstIterator<uint8_t>(*msg, name);
                break;
            case sensor_msgs::msg::PointField::INT16:
                it_int16 = sensor_msgs::PointCloud2ConstIterator<int16_t>(*msg, name);
                break;
            case sensor_msgs::msg::PointField::UINT16:
                it_uint16 = sensor_msgs::PointCloud2ConstIterator<uint16_t>(*msg, name);
                break;
            case sensor_msgs::msg::PointField::INT32:
                it_int32 = sensor_msgs::PointCloud2ConstIterator<int32_t>(*msg, name);
                break;
            case sensor_msgs::msg::PointField::UINT32:
                it_uint32 = sensor_msgs::PointCloud2ConstIterator<uint32_t>(*msg, name);
                break;
            case sensor_msgs::msg::PointField::FLOAT32:
                it_float = sensor_msgs::PointCloud2ConstIterator<float>(*msg, name);
                break;
            case sensor_msgs::msg::PointField::FLOAT64:
                it_double = sensor_msgs::PointCloud2ConstIterator<double>(*msg, name);
                break;
            default: assert(false && "unsupported field type"); 
            };
        }
    };

    union Scalar {
        int8_t val_int8;
        uint8_t val_uint8;
        int16_t val_int16;
        uint16_t val_uint16;
        int32_t val_int32;
        uint32_t val_uint32;
        float val_float;
        double val_double;

        Scalar() noexcept = default;

        template<typename T>
        Scalar(T val) noexcept {
            if constexpr (std::is_same_v<T, int8_t>)        val_int8 = val;        
            else if constexpr (std::is_same_v<T, uint8_t>)  val_uint8 = val;  
            else if constexpr (std::is_same_v<T, int16_t>)  val_int16 = val;  
            else if constexpr (std::is_same_v<T, uint16_t>) val_uint16 = val; 
            else if constexpr (std::is_same_v<T, int32_t>)  val_int32 = val;  
            else if constexpr (std::is_same_v<T, uint32_t>) val_uint32 = val; 
            else if constexpr (std::is_same_v<T, float>)    val_float = val;
            else if constexpr (std::is_same_v<T, double>)   val_double = val;
            else static_assert(sizeof(T) == 0, "unsupported field type");
        }

        Scalar(ScalarConstIterator& it, uint8_t datatype) noexcept {
            auto take = [](auto& iter) {
                auto val = *iter;
                ++iter;
                return val;
            };

            switch(datatype) {
            case sensor_msgs::msg::PointField::INT8:
                val_int8 = take(it.it_int8);
                break;
            case sensor_msgs::msg::PointField::UINT8:
                val_uint8 = take(it.it_uint8);
                break;
            case sensor_msgs::msg::PointField::INT16:
                val_int16 = take(it.it_int16);
                break;
            case sensor_msgs::msg::PointField::UINT16:
                val_uint16 = take(it.it_uint16);
                break;
            case sensor_msgs::msg::PointField::INT32:
                val_int32 = take(it.it_int32);
                break;
            case sensor_msgs::msg::PointField::UINT32:
                val_uint32 = take(it.it_uint32);
                break;
            case sensor_msgs::msg::PointField::FLOAT32:
                val_float = take(it.it_float);
                break;
            case sensor_msgs::msg::PointField::FLOAT64:
                val_double = take(it.it_double);
                break;
            default: assert(false && "unsupported field type");  
            };
        }

        template<typename T>
        T get() const noexcept {
            if constexpr (std::is_same_v<T, int8_t>)       return val_int8;
            else if constexpr (std::is_same_v<T, uint8_t>) return val_uint8;
            else if constexpr (std::is_same_v<T, int16_t>) return val_int16;
            else if constexpr (std::is_same_v<T, uint16_t>)return val_uint16;
            else if constexpr (std::is_same_v<T, int32_t>) return val_int32;
            else if constexpr (std::is_same_v<T, uint32_t>)return val_uint32;
            else if constexpr (std::is_same_v<T, float>)   return val_float;
            else if constexpr (std::is_same_v<T, double>)  return val_double;
            else static_assert(sizeof(T) == 0, "unsupported field type");
        }

        void write(ScalarIterator& it, uint8_t datatype) const noexcept {
            switch(datatype) {
            case sensor_msgs::msg::PointField::INT8:
                *(it.it_int8) = val_int8;
                ++(it.it_int8);
                break;
            case sensor_msgs::msg::PointField::UINT8:
                *(it.it_uint8) = val_uint8;
                ++(it.it_uint8);
                break;
            case sensor_msgs::msg::PointField::INT16:
                *(it.it_int16) = val_int16;
                ++(it.it_int16);
                break;
            case sensor_msgs::msg::PointField::UINT16:
                *(it.it_uint16) = val_uint16;
                ++(it.it_uint16);
                break;
            case sensor_msgs::msg::PointField::INT32:
                *(it.it_int32) = val_int32;
                ++(it.it_int32);
                break;
            case sensor_msgs::msg::PointField::UINT32:
                *(it.it_uint32) = val_uint32;
                ++(it.it_uint32);
                break;
            case sensor_msgs::msg::PointField::FLOAT32:
                *(it.it_float) = val_float;
                ++(it.it_float);
                break;
            case sensor_msgs::msg::PointField::FLOAT64:
                *(it.it_double) = val_double;
                ++(it.it_double);
                break;
            default: assert(false && "unsupported field type");  
            };
        }
    };

    CloudDescriptor cloud_desc_;
    std::vector<std::vector<Scalar>> field_scalars_;
};
}
