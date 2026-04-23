#pragma once

#include <vector>
#include <map>

#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Dense>

#include "aliases.h"

EigenCloudWithIntensity pointcloud2_with_intensity_to_eigen(const sensor_msgs::PointCloud2ConstPtr& msg);

sensor_msgs::PointCloud2Ptr pointcloud2_from_eigen(const std_msgs::Header& header, 
    const EigenCloud& points, const std::map<std::string, EigenCloudField> fields = {});