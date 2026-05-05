#pragma once
// pybind
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/eigen.h>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <Eigen/Dense>

namespace py = pybind11;

// TODO: I would like to automatically generate these bindings somehow
// for every type in sensor_msgs::msg::PointField
using VectorINT8 = std::vector<int8_t>;
using VectorUINT8 = std::vector<uint8_t>;
using VectorINT16 = std::vector<int16_t>;
using VectorUINT16 = std::vector<uint16_t>;
using VectorINT32 = std::vector<int32_t>;
using VectorUINT32 = std::vector<uint32_t>;
using VectorFLOAT32 = std::vector<float>;
using VectorFLOAT64 = std::vector<double>;

PYBIND11_MAKE_OPAQUE(VectorINT8);
PYBIND11_MAKE_OPAQUE(VectorUINT8);
PYBIND11_MAKE_OPAQUE(VectorINT16);
PYBIND11_MAKE_OPAQUE(VectorUINT16);
PYBIND11_MAKE_OPAQUE(VectorINT32);
PYBIND11_MAKE_OPAQUE(VectorUINT32);
PYBIND11_MAKE_OPAQUE(VectorFLOAT32);
PYBIND11_MAKE_OPAQUE(VectorFLOAT64);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3f>);
