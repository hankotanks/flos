#pragma once

#include <vector>
#include <variant>
#include <map>

#include <Eigen/Dense>

// these are defined indirectly so that we can automatically generate the repr with a stringify macro
#define POINT_TYPE Eigen::Matrix<float, 3, 1>
#define CLOUD_TYPE std::vector<POINT_TYPE>

using Point = POINT_TYPE;
using FieldF32 = std::vector<float>;
using FieldU8 = std::vector<uint8_t>;
using Field = std::variant<FieldF32, FieldU8>;
using Cloud = CLOUD_TYPE;
using CloudWithIntensity = std::pair<Cloud, FieldF32>;
using CloudWithFields = std::pair<Cloud, std::map<std::string, Field>>;