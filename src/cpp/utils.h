#pragma once

#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Dense>

std::vector<Eigen::Vector3f> point_cloud2_to_eigen(const sensor_msgs::PointCloud2::ConstPtr& msg);