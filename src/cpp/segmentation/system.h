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
        std::set<size_t> ignore{0, 2, 4}; // NOTE: by default, ignore shrubs and canopy
    };
public:
    class Model {
        friend SegmentationSystem;
    public:
        explicit Model(const SegmentationSystem::Config& config, const std::string& path_to_model);
    public:
        void infer(util::Cloud& cloud, bool preprocessed = false);
    public:
        Model() = delete;
        Model(const Model&) = delete;
        Model(Model&&) = delete;
        Model& operator=(const Model&) = delete;
        Model& operator=(Model&&) = delete;
        ~Model() = default;
    private:
        py::object module_;
        py::object helper_;
        py::object infer_;
        SegmentationSystem::Config config_;
    };
public:
    explicit SegmentationSystem(const Config& config) : config_(config) { /* STUB */ }
public:
    void preprocess(util::Cloud& cloud);
    SegmentationSystem::Model load(const std::string& path_to_model);
public:
    SegmentationSystem() = delete;
    SegmentationSystem(const SegmentationSystem&) = delete;
    SegmentationSystem(SegmentationSystem&&) = delete;
    SegmentationSystem& operator=(const SegmentationSystem&) = delete;
    SegmentationSystem& operator=(SegmentationSystem&&) = delete;
    ~SegmentationSystem() = default;
public:
    // NOTE: Config::voxel_size is scaled by this constant for unified parameter passing between segmentation and odometry
    static constexpr double VOXEL_SIZE_SCALAR{0.5};
    // scalar for map distance
    // NOTE: the idea is that we want the map to be larger than any incoming scan's max distance
    static constexpr double MAX_DISTANCE_SCALAR{1.0};
protected:
    // system configuration
    Config config_;    
};