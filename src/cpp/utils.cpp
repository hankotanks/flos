#include "utils.h"

#include <regex>
#include <map>
#include <sensor_msgs/point_cloud2_iterator.h>

namespace {
inline std::string FixFrameId(const std::string& frame_id) {
  return std::regex_replace(frame_id, std::regex("^/"), "");
}
} // namespace


std::pair<std::vector<Eigen::Matrix<float, 3, 1>>, std::vector<float>>
point_cloud2_to_eigen_with_intensities(const sensor_msgs::PointCloud2ConstPtr& msg) {
  const size_t point_count = static_cast<size_t>(msg->height) * msg->width;
  std::vector<Eigen::Matrix<float, 3, 1>> points;
  std::vector<float> intensities;
  points.reserve(point_count);
  intensities.reserve(point_count);
  sensor_msgs::PointCloud2ConstIterator<float> msg_x(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> msg_y(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> msg_z(*msg, "z");
  sensor_msgs::PointCloud2ConstIterator<float> msg_intensity(*msg, "intensity");
  for (size_t i = 0; i < point_count; ++i, ++msg_x, ++msg_y, ++msg_z, ++msg_intensity) {
    points.emplace_back(*msg_x, *msg_y, *msg_z);
    intensities.emplace_back(*msg_intensity);
  }
  return {points, intensities};
}

sensor_msgs::PointCloud2Ptr 
eigen_to_point_cloud2_with_fields(const std::vector<Eigen::Matrix<float, 3, 1>>& points, 
    const std::map<std::string, std::vector<float>> fields, const std_msgs::Header& header) {

    // Create PointCloud2 message with intensity field
    auto cloud_msg = std::make_unique<sensor_msgs::PointCloud2>();
    sensor_msgs::PointCloud2Modifier modifier(*cloud_msg);
    cloud_msg->header = header;
    cloud_msg->header.frame_id = FixFrameId(cloud_msg->header.frame_id);
    cloud_msg->fields.clear();

    int offset = 0;
    offset = addPointField(*cloud_msg, "x", 1, sensor_msgs::PointField::FLOAT32, offset);
    offset = addPointField(*cloud_msg, "y", 1, sensor_msgs::PointField::FLOAT32, offset);
    offset = addPointField(*cloud_msg, "z", 1, sensor_msgs::PointField::FLOAT32, offset);
    for(const auto& [field, field_data] : fields) {
        offset = addPointField(*cloud_msg, field, 1, sensor_msgs::PointField::FLOAT32, offset);
    }

    cloud_msg->point_step = offset;
    cloud_msg->row_step = cloud_msg->width * cloud_msg->point_step;
    cloud_msg->data.resize(static_cast<size_t>(cloud_msg->height) * cloud_msg->row_step);

    modifier.resize(points.size());

    // Fill the data
    sensor_msgs::PointCloud2Iterator<float> msg_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> msg_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> msg_z(*cloud_msg, "z");

    std::map<std::string, sensor_msgs::PointCloud2Iterator<float>> msg_fields;
    for(const auto& [field, field_data] : fields) {
      msg_fields.emplace(std::piecewise_construct,
                   std::forward_as_tuple(field),
                   std::forward_as_tuple(*cloud_msg, field));
    }

    for (size_t i = 0; i < points.size(); ++i, ++msg_x, ++msg_y, ++msg_z) {
        const Eigen::Vector3f& point = points[i];
        *msg_x = static_cast<float>(point.x());
        *msg_y = static_cast<float>(point.y());
        *msg_z = static_cast<float>(point.z());
        for(const auto& [field, field_data] : fields) {
          *msg_fields.at(field) = static_cast<float>(field_data[i]);
          ++msg_fields.at(field);
        }   
    }

    return cloud_msg;
}
