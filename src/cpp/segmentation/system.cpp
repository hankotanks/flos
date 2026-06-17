#include "system.h"

// util
#include "util/voxel.h"

void SegmentationSystem::initialize(const std::string& path_to_model) {
    module_ = py::module_::import("segmentation_model.inference_helper");
    helper_ = module_.attr("InferenceHelper")(path_to_model);
    infer_ = helper_.attr("infer");
}

void SegmentationSystem::preprocess(util::Cloud& cloud) {
}

void SegmentationSystem::infer(util::Cloud& cloud, bool preprocessed) {
    // conditionally preprocess
    if(!preprocessed) SegmentationSystem::preprocess(cloud);
    py::object points = cloud.points_as_py_object();
    py::object intensity = cloud.field_as_py_object<float>("intensity");
    // perform inference
    py::object result = infer_(points, intensity).cast<py::tuple>();
    if(!py::isinstance<py::tuple>(result)) throw std::runtime_error("inference failed");
    py::tuple result_fields = result.cast<py::tuple>();
    // output
    cloud.add_field_from_py_object<uint8_t>("class", result_fields[0]);
    cloud.add_field_from_py_object<float>("confidence", result_fields[1]);
    cloud.filter<uint8_t>("class", [this](const uint8_t& scalar) {
        return (scalar != 0) && !(this->config_.ignore_shrubs && scalar == 2) && !(this->config_.ignore_canopy && scalar == 4);
    });
}
