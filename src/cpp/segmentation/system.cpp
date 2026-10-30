#include "segmentation/system.h"

// system
#include <stdexcept>
// tbb
#include <tbb/parallel_for.h>
// package
#include "classes.h"
// util
#include "util/profiler.h"

namespace {
py::object load_from_segmentation_model(const std::string& name) {
    std::filesystem::path path_to_source = std::filesystem::path{PROJECT_ROOT};
    std::filesystem::path path_to_module = path_to_source.parent_path() / "python";
    py::module_::import("sys").attr("path").attr("insert")(0, path_to_module.string());
    return py::module_::import(("segmentation_model." + name).c_str());
}

void predict(std::span<uint8_t> prediction, const std::span<std::span<const float>> fields) {
    if(fields.empty()) return;
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, prediction.size()), [&prediction, &fields](const tbb::blocked_range<std::size_t>& range) {
        for(size_t i = range.begin(); i != range.end(); ++i) {
            float best_prob = fields[0][i];
            size_t best_pred = 0;
            for(size_t pred = 1; pred < fields.size(); ++pred) {
                const float& prob = fields[pred][i];
                if(prob > best_prob) {
                    best_prob = prob;
                    best_pred = pred;
                }
            }
            prediction[i] = static_cast<uint8_t>(best_pred);
        }
    });
}
} // private scope

SegmentationSystem::SegmentationSystem(const Config& config) : SegmentationSystem(config, nullptr) { /* STUB */ }
SegmentationSystem::SegmentationSystem(const Config& config, util::Profiler* profiler) : 
    util::Profiler("SegmentationSystem", profiler), config_(config) { /* STUB */ }

SegmentationSystem::Model::Model(const SegmentationSystem::Config& config, const std::filesystem::path& path_to_model, util::Profiler* profiler) :
    profiler_(profiler),
    config_(config),
    module_(load_from_segmentation_model("inference_helper")),
    helper_(module_.attr("InferenceHelper")(path_to_model.string())),
    infer_(helper_.attr("infer")) { /* STUB */ }

void SegmentationSystem::preprocess(util::Cloud& cloud) {
    cloud.downsample(config_.voxel_size_inference);
    cloud.filter_by_distance(Eigen::Vector3f::Zero(), 0.f, config_.max_distance);
}

SegmentationSystem::Model SegmentationSystem::load(const std::filesystem::path& path_to_model) {
    return SegmentationSystem::Model{config_, path_to_model, this};
}

void SegmentationSystem::Model::infer(util::Cloud& cloud) {
    PROFILE(inference, profiler_) {
        py::object points = cloud.points_as_py_object();
        std::vector<std::span<const float>> fields;
        fields.reserve(classes::COUNT);
        try {
            // perform inference
            py::list result;
            PROFILE(python, &inference) {
                result = infer_(points).cast<py::list>();
            }
            // output
            for(size_t i = 0; i < result.size(); ++i) {
                fields.emplace_back(cloud.add_field_from_py_object<float>(std::to_string(i), result[i]));
            }
        } catch(const py::cast_error&) {
            throw std::runtime_error("inference failed");
        }
        // postprocessing
        PROFILE(postprocessing, &inference) {
            // argmax
            std::span<uint8_t> prediction = cloud.add_field<uint8_t>("prediction", 0);
            predict(prediction, fields);
            // remap semantic classes...
            if(!config_.semantics.remap.empty()) {
                cloud.transform<uint8_t>("prediction", [this](const uint8_t& prediction) {
                    if(this->config_.semantics.remap.contains(prediction)) {
                        return static_cast<uint8_t>(this->config_.semantics.remap.at(static_cast<size_t>(prediction)));
                    } else {
                        return prediction;
                    }
                });
                // ...and accumulate their respective probabilities
                for(const auto& [prediction_from, prediction_to] : config_.semantics.remap) {
                    std::span<float> from = cloud.field<float>(std::to_string(prediction_from));
                    std::span<float> to = cloud.field<float>(std::to_string(prediction_to));
                    std::transform(to.begin(), to.end(), from.begin(), to.begin(), std::plus<>{});
                    std::fill(from.begin(), from.end(), 0.f);
                }
            }
            // remove ignored classes
            cloud.filter<uint8_t>("prediction", [this](const uint8_t& scalar) {
                return this->config_.semantics.ignore.count(scalar) == 0;
            });
            // downsample only if the inference resolution was different from the output resolution
            float voxel_size = config_.double_downsample ? config_.voxel_size * SegmentationSystem::VOXEL_SIZE_SCALAR : config_.voxel_size;
            if(std::abs(voxel_size - config_.voxel_size_inference) > 0.01f) {
                cloud.downsample(voxel_size);
            }
        }
    }
}

void SegmentationSystem::Model::preprocess(util::Cloud& cloud) {
    cloud.downsample(config_.voxel_size_inference);
    cloud.filter_by_distance(Eigen::Vector3f::Zero(), 0.f, config_.max_distance);
}

std::ostream& operator<<(std::ostream& stream, const SegmentationSystem::Config& config) {
    auto format_ignore = [&stream](const std::set<size_t>& ignore) -> std::string {
        std::ostringstream oss;
        oss << '[';
        bool first = true;
        for(size_t i : ignore) {
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

    auto format_remap = [&stream](const std::map<size_t, size_t>& remap) -> std::string {
        std::ostringstream oss;
        oss << '[';
        bool first = true;
        for(const auto [i, w] : remap) {
            if(!first) oss << ", ";
            first = false; 
            if(i < classes::COUNT) {
                oss << classes::HINT[i];
            } else {
                oss << i;
            }
            oss << ": ";
            if(w < classes::COUNT) {
                oss << classes::HINT[w];
            } else {
                oss << w;
            }
        }
        oss << ']';
        return oss.str();
    };

    return stream << std::boolalpha << "SegmentationSystem::Config" << std::endl <<
        "\tmax_distance: " << config.max_distance << "," << std::endl << 
        "\tvoxel_size: " << config.voxel_size << "," << std::endl << 
        "\tvoxel_size_inference: " << config.voxel_size_inference << "," << std::endl << 
        "\tdouble_downsample: " << config.double_downsample << "," << std::endl <<
        "\tsemantics_ignore: " << format_ignore(config.semantics.ignore) << "," << std::endl <<
        "\tsemantics_remap: " << format_remap(config.semantics.remap);
}