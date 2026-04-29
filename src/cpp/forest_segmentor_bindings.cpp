#include "forest_segmentor_bindings.h"

#include "stl_vector_eigen.h"

#define STRINGIFY_INNER(...) #__VA_ARGS__
#define STRINGIFY(...) STRINGIFY_INNER(__VA_ARGS__)

// binding code here still works because it just needs to be visible in the final binary
PYBIND11_EMBEDDED_MODULE(forest_segmentor_bindings, m) {
    // this one still copies i believe for some reason
    py::bind_vector<FieldF32>(m, "FieldF32");
    py::bind_vector<FieldU8>(m, "FieldU8");
    // (n, 3) numpy array
    pybind_eigen_vector_of_vector<Point>(m, "Cloud", 
        STRINGIFY(POINT_TYPE), py::py_array_to_vectors<Eigen::Vector3f>);
}
