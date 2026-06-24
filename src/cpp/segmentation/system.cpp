#include "system.h"

// util
#include "util/voxel.h"
#include <stdexcept>

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
    try {
        // perform inference
        py::tuple result = infer_(points, intensity).cast<py::tuple>();
        // output
        cloud.add_field_from_py_object<uint8_t>("class", result[0]);
        py::list result_confidence = result[1].cast<py::list>();
        for(size_t i = 0; i < result_confidence.size(); ++i) {
            if(this->config_.ignore.count(i) > 0) continue;
            cloud.add_field_from_py_object<float>(std::to_string(i), result_confidence[i]);
        }
    } catch(const py::cast_error&) {
        throw std::runtime_error("inference failed");
    }
    // TODO: do we still perform the belief update with points of ignored classes?
    cloud.filter<uint8_t>("class", [this](const uint8_t& scalar) {
        return this->config_.ignore.count(scalar) == 0;
    });
}
