#include "forest_segmentor_bindings.h"
#include "stl_vector_eigen.h"

// binding code here still works because it just needs to be visible in the final binary
PYBIND11_EMBEDDED_MODULE(forest_segmentor_bindings, m) {
    pybind_eigen_vector_of_vector<Eigen::Vector3f>(m, "EigenCloud", "std::vector<Eigen::Matrix<float, 3, 1>>", 
        py::py_array_to_vectors<Eigen::Vector3f>);
    // this one still copies i believe for some reason
    py::bind_vector<std::vector<float>>(m, "EigenCloudField");
}
