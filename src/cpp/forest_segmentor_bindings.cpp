#include "forest_segmentor_bindings.h"

#include "stl_vector_eigen.h"

// binding code here still works because it just needs to be visible in the final binary
PYBIND11_EMBEDDED_MODULE(forest_segmentor_bindings, m) {
    // this one still copies i believe for some reason
    py::bind_vector<VectorINT8>(m, "VectorINT8");
    py::bind_vector<VectorUINT8>(m, "VectorUINT8");
    py::bind_vector<VectorINT16>(m, "VectorINT16");
    py::bind_vector<VectorUINT16>(m, "VectorUINT16");
    py::bind_vector<VectorINT32>(m, "VectorINT32");
    py::bind_vector<VectorUINT32>(m, "VectorUINT32");
    py::bind_vector<VectorFLOAT32>(m, "VectorFLOAT32");
    py::bind_vector<VectorFLOAT64>(m, "VectorFLOAT64");
    // (n, 3) numpy array
    pybind_eigen_vector_of_vector<Eigen::Vector3f>(m, "Cloud", "Cloud", py::py_array_to_vectors<Eigen::Vector3f>);
}
