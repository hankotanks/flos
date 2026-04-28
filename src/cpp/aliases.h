#pragma once

#include <vector>

#include <Eigen/Dense>

using EigenCloud = std::vector<Eigen::Matrix<float, 3, 1>>;
using EigenCloudField = std::vector<float>;
using EigenCloudWithIntensity = std::pair<EigenCloud, EigenCloudField>;