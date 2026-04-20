#include "utils.h"

#include <sensor_msgs/point_cloud2_iterator.h>

std::vector<Eigen::Vector3f> point_cloud2_to_eigen(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    const size_t point_count = static_cast<size_t>(msg->height) * msg->width;
    std::vector<Eigen::Vector3f> points;
    points.reserve(point_count);
    sensor_msgs::PointCloud2ConstIterator<float> msg_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> msg_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> msg_z(*msg, "z");
    for (size_t i = 0; i < point_count; ++i, ++msg_x, ++msg_y, ++msg_z) {
        points.emplace_back(*msg_x, *msg_y, *msg_z);
    }
    return points;
}