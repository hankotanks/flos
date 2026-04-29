#include "utils.h"
#include "aliases.h"

#include <regex>
#include <map>
#include <any>

#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace {
    inline std::string fix_frame_id(const std::string &frame_id) {
        return std::regex_replace(frame_id, std::regex("^/"), "");
    }
} // private scope

CloudWithIntensity pointcloud2_with_intensity_to_eigen(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
    const size_t point_count = static_cast<size_t>(msg->height) * msg->width;
    std::vector<Eigen::Matrix<float, 3, 1>> points;
    std::vector<float> intensities;
    points.reserve(point_count);
    intensities.reserve(point_count);
    sensor_msgs::PointCloud2ConstIterator<float> msg_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> msg_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> msg_z(*msg, "z");
    sensor_msgs::PointCloud2ConstIterator<float> msg_intensity(*msg, "intensity");
    for(size_t i = 0; i < point_count; ++i, ++msg_x, ++msg_y, ++msg_z, ++msg_intensity) {
        points.emplace_back(*msg_x, *msg_y, *msg_z);
        intensities.emplace_back(*msg_intensity);
    }
    return std::make_pair(points, intensities);
}

sensor_msgs::msg::PointCloud2 pointcloud2_from_eigen(const std_msgs::msg::Header& header, 
    const Cloud& points, const std::map<std::string, Field>& fields) {
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
    for (const auto& [field_name, field_data] : fields) {
        std::visit([&](auto&& vec){
            using T = std::decay_t<decltype(vec)>;
            int type;
            if constexpr (std::is_same_v<T, std::vector<float>>) type = sensor_msgs::msg::PointField::FLOAT32;
            else if constexpr (std::is_same_v<T, std::vector<uint8_t>>) type = sensor_msgs::msg::PointField::UINT8;
            offset = addPointField(msg, field_name, 1, type, offset);
        }, field_data);
    }
    msg.point_step = offset;
    msg.row_step = msg.width * msg.point_step;
    msg.data.resize(static_cast<size_t>(msg.height) * msg.row_step);
    modifier.resize(points.size());
    // position iterators
    sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");
    // build iterators for each field
    std::map<std::string, std::any> msg_fields;
    for(const auto& [field_name, field_data] : fields) {
        std::visit([&](auto&& vec){
            using T = std::decay_t<decltype(vec)>;
            msg_fields[field_name] = sensor_msgs::PointCloud2Iterator<typename T::value_type>(msg, field_name);
        }, field_data);
    }
    // populate the message
    for(size_t i = 0; i < points.size(); ++i, ++msg_x, ++msg_y, ++msg_z) {
        const Eigen::Vector3f& point = points[i];
        *msg_x = point.x();
        *msg_y = point.y();
        *msg_z = point.z();
        for(const auto& [field_name, field_data] : fields) {
            std::visit([&](auto&& vec){
                using T = std::decay_t<decltype(vec)>;
                auto& it = std::any_cast<sensor_msgs::PointCloud2Iterator<typename T::value_type>&>(msg_fields[field_name]);
                *it = vec[i];
                ++it;
            }, field_data);
        }
    }
    return msg;
}