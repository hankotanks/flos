#pragma once

#include <map>

#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <Eigen/Dense>

#include "aliases.h"

CloudWithIntensity pointcloud2_with_intensity_to_eigen(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg);

sensor_msgs::msg::PointCloud2 pointcloud2_from_eigen(const std_msgs::msg::Header& header, 
    const Cloud& points, const std::map<std::string, Field>& fields = {});