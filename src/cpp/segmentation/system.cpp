#include "segmentation/system.h"

// system
#include <stdexcept>
// util
#include "util/profiler.h"
#include "util/voxel.h"

SegmentationSystem::Model::Model(const SegmentationSystem::Config& config, const std::filesystem::path& path_to_model, util::Profiler* profiler) :
    util::Profiler("Model", profiler),
    config_(config),
    module_(py::module_::import("segmentation_model.inference_helper")),
    helper_(module_.attr("InferenceHelper")(path_to_model.string())),
    infer_(helper_.attr("infer")) { /* STUB */ }

void preprocess_inner(const SegmentationSystem::Config& config, util::Cloud& cloud) {
    double voxel_size = config.voxel_size * SegmentationSystem::VOXEL_SIZE_SCALAR_INFERENCE;
    util::voxel_downsample(cloud, voxel_size);
    double max_distance = config.max_distance * SegmentationSystem::MAX_DISTANCE_SCALAR;
    cloud.filter_by_distance(Eigen::Vector3f::Zero(), 0.f, max_distance);
}

void SegmentationSystem::preprocess(util::Cloud& cloud) {
    util::Profiler profiler{"preprocess", this};
    preprocess_inner(config_, cloud);
}

SegmentationSystem::Model SegmentationSystem::load(const std::filesystem::path& path_to_model) {
    return SegmentationSystem::Model{config_, path_to_model, this};
}

void SegmentationSystem::Model::infer(util::Cloud& cloud, bool preprocessed) {
    util::Profiler profiler{"infer", this};
    // conditionally preprocess
    if(!preprocessed) {
        util::Profiler subprofiler{"preprocess", &profiler};
        preprocess_inner(config_, cloud);
    }
    py::object points = cloud.points_as_py_object();
    py::object intensity = cloud.field_as_py_object<float>("intensity");
    try {
        // perform inference
        py::tuple result;
        {  
            util::Profiler subprofiler{"python", &profiler};
            result = infer_(points, intensity).cast<py::tuple>();
        }
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
    cloud.filter<uint8_t>("class", [this](const uint8_t& scalar) {
        return this->config_.ignore.count(scalar) == 0;
    });
    // TODO: in progress
    cloud.filter_by_distance<uint8_t>(Eigen::Vector3f::Zero(), 0.f, config_.max_distance_neighborhood, "class", [this](const uint8_t& scalar) -> bool {
        return this->config_.nearby.count(scalar) > 0;
    });
    if(std::abs(SegmentationSystem::VOXEL_SIZE_SCALAR - SegmentationSystem::VOXEL_SIZE_SCALAR_INFERENCE) < 0.01) return;
    double voxel_size = config_.voxel_size * SegmentationSystem::VOXEL_SIZE_SCALAR;
    util::voxel_downsample(cloud, voxel_size);
}

std::ostream& operator<<(std::ostream& stream, const SegmentationSystem::Config& config) {
    auto format = [&stream](const std::set<size_t>& indices) -> std::string {
        std::ostringstream oss;
        oss << '[';
        bool first = true;
        for(size_t i : indices) {
            if(!first) oss << ", ";
            first = false; 
            if(i < SegmentationSystem::Config::CLASS_HINT.size()) {
                oss << SegmentationSystem::Config::CLASS_HINT[i];
            } else {
                oss << i;
            }
        }
        oss << ']';
        return oss.str();
    };

    return stream << std::boolalpha << "SegmentationSystem::Config" << std::endl <<
        "\tvoxel_size: " << config.voxel_size << "," << std::endl << 
        "\tmax_distance: " << config.max_distance << "," << std::endl << 
        "\tmax_distance_neighborhood: " << config.max_distance_neighborhood << "," << std::endl << 
        "\tignore: " << format(config.ignore) << "," << std::endl << 
        "\tnearby: " << format(config.nearby);
}