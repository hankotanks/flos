#include "segmentation/system.h"

// system
#include <stdexcept>
// package
#include "classes.h"
// util
#include "util/profiler.h"

SegmentationSystem::Model::Model(const SegmentationSystem::Config& config, const std::filesystem::path& path_to_model, util::Profiler* profiler) :
    profiler_(profiler),
    config_(config),
    module_(py::module_::import("segmentation_model.inference_helper")),
    helper_(module_.attr("InferenceHelper")(path_to_model.string())),
    infer_(helper_.attr("infer")) { /* STUB */ }

void preprocess_inner(const SegmentationSystem::Config& config, util::Cloud& cloud) {
    double voxel_size = config.voxel_size * SegmentationSystem::VOXEL_SIZE_SCALAR_INFERENCE;
    cloud.downsample(voxel_size);
    double max_distance = config.max_distance * SegmentationSystem::MAX_DISTANCE_SCALAR;
    cloud.filter_by_distance(Eigen::Vector3f::Zero(), 0.f, max_distance);
}

void SegmentationSystem::preprocess(util::Cloud& cloud) {
    PROFILE(preprocessing, this) {
        preprocess_inner(config_, cloud);
    }
}

SegmentationSystem::Model SegmentationSystem::load(const std::filesystem::path& path_to_model) {
    return SegmentationSystem::Model{config_, path_to_model, this};
}

void SegmentationSystem::Model::infer(util::Cloud& cloud) {
    PROFILE(inference, profiler_) {
        py::object points = cloud.points_as_py_object();
        try {
            // perform inference
            py::tuple result;
            PROFILE(python, &inference) {
                result = infer_(points).cast<py::tuple>();
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
        PROFILE(postprocessing, &inference) {
            // remove ignored classes
            cloud.filter<uint8_t>("class", [this](const uint8_t& scalar) {
                return this->config_.ignore.count(scalar) == 0;
            });
            // downsample only if the inference resolution was different from the output resolution
            if(std::abs(SegmentationSystem::VOXEL_SIZE_SCALAR - SegmentationSystem::VOXEL_SIZE_SCALAR_INFERENCE) > 0.01) {
                double voxel_size = config_.voxel_size * SegmentationSystem::VOXEL_SIZE_SCALAR;
                cloud.downsample(voxel_size);
            }
        }
    }
}

void SegmentationSystem::Model::preprocess(util::Cloud& cloud) {
    PROFILE(preprocessing, profiler_) {
        preprocess_inner(config_, cloud);
    }
}

std::ostream& operator<<(std::ostream& stream, const SegmentationSystem::Config& config) {
    auto format = [&stream](const std::set<size_t>& indices) -> std::string {
        std::ostringstream oss;
        oss << '[';
        bool first = true;
        for(size_t i : indices) {
            if(!first) oss << ", ";
            first = false; 
            if(i < classes::COUNT) {
                oss << classes::HINT[i];
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
        "\tignore: " << format(config.ignore);
}