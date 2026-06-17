#pragma once

// pybind
#include <pybind11/embed.h>
// util
#include "util/cloud.h"

class SegmentationSystem final {
public:
    struct Config {
        double voxel_size{1.0};
        double max_distance{25.0};
        bool ignore_canopy{true};
        bool ignore_shrubs{true};
    };
public:
    explicit SegmentationSystem(const Config& config) : config_(config) { /* STUB */ }
public:
    // system must be initialized before inferencing occurs
    void initialize(const std::string& path_to_model);
    void preprocess(util::Cloud& cloud);
    void infer(util::Cloud& cloud, bool preprocessed = false);
public:
    SegmentationSystem() = delete;
    SegmentationSystem(const SegmentationSystem&) = delete;
    SegmentationSystem(SegmentationSystem&&) = delete;
    SegmentationSystem& operator=(const SegmentationSystem&) = delete;
    SegmentationSystem& operator=(SegmentationSystem&&) = delete;
    ~SegmentationSystem() = default;
private:
    // NOTE: Config::voxel_size is scaled by this constant for unified parameter passing between segmentation and odometry
    static constexpr double VOXEL_SIZE_SCALAR{0.5};
    // scalar for map distance
    // NOTE: the idea is that we want the map to be larger than any incoming scan's max distance
    static constexpr double MAX_DISTANCE_SCALAR{0.75};
    // system configuration
    Config config_;
    // bindings
    py::object module_;
    py::object helper_;
    py::object infer_;
};