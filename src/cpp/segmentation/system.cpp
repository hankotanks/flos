#include "system.h"

// util
#include "util/voxel.h"

SegmentationSystem::Model::Model(const SegmentationSystem::Config& config, const std::string& path_to_model) :
    config_(config),
    module_(py::module_::import("segmentation_model.inference_helper")),
    helper_(module_.attr("InferenceHelper")(path_to_model)),
    infer_(helper_.attr("infer")) { /* STUB */ }

void preprocess_inner(const SegmentationSystem::Config& config, util::Cloud& cloud) {
    double voxel_size = config.voxel_size * SegmentationSystem::VOXEL_SIZE_SCALAR;
    util::voxel_downsample(cloud, voxel_size);
    double max_distance = config.max_distance * SegmentationSystem::MAX_DISTANCE_SCALAR;
    cloud.filter_by_distance(Eigen::Vector3f::Zero(), max_distance);
}

void SegmentationSystem::preprocess(util::Cloud& cloud) {
    preprocess_inner(config_, cloud);
}

SegmentationSystem::Model SegmentationSystem::load(const std::string& path_to_model) {
    return SegmentationSystem::Model{config_, path_to_model};
}

void SegmentationSystem::Model::infer(util::Cloud& cloud, bool preprocessed) {
    // conditionally preprocess
    if(!preprocessed) preprocess_inner(config_, cloud);
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
