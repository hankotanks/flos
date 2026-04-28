#pragma once
// pybind
#include <pybind11/eigen.h>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <Eigen/Dense>

#include "aliases.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(EigenCloud);
PYBIND11_MAKE_OPAQUE(EigenCloudField);