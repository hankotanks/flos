#pragma once

#include <vector>
#include <map>

#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Dense>

std::pair<std::vector<Eigen::Matrix<float, 3, 1>>, std::vector<float>> 
    point_cloud2_to_eigen_with_intensities(const sensor_msgs::PointCloud2ConstPtr& msg);

sensor_msgs::PointCloud2Ptr 
eigen_to_point_cloud2_with_fields(const std::vector<Eigen::Matrix<float, 3, 1>>& points, 
    const std::map<std::string, std::vector<float>> fields, const std_msgs::Header& header);