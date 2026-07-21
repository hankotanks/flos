#pragma once

// system
#include <filesystem>
// pybind
#include <pybind11/embed.h>
// util
#include "util/cloud.h"
#include "util/profiler.h"

class SegmentationSystem : util::Profiler {
public:
    struct Config {
        static constexpr std::array<std::string, 5> CLASS_HINT{"ignore", "ground", "shrubs", "trunks", "canopy"};
        float voxel_size{1.f};
        float max_distance{MAX_DISTANCE_DEFAULT};
        float max_distance_neighborhood{MAX_DISTANCE_DEFAULT * 0.5f};
        std::set<size_t> nearby; // NOTE: by default, limit ground to local_distance
        std::set<size_t> ignore{0, 4}; // NOTE: by default, ignore canopy
    public:
        friend std::ostream& operator<<(std::ostream& stream, const Config& config);
    private:
        static constexpr float MAX_DISTANCE_DEFAULT{50.f};
    };
public:
    class Model {
        friend SegmentationSystem;
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
        explicit Model(const SegmentationSystem::Config& config, const std::filesystem::path& path_to_model, util::Profiler* profiler);
    private:
        util::Profiler* profiler_;
        py::object module_;
        py::object helper_;
        py::object infer_;
        SegmentationSystem::Config config_;
    };
public:
    explicit SegmentationSystem(const Config& config) : util::Profiler("SegmentationSystem"), config_(config) { /* STUB */ }
public:
    void preprocess(util::Cloud& cloud);
    SegmentationSystem::Model load(const std::filesystem::path& path_to_model);
public:
    SegmentationSystem() = delete;
    SegmentationSystem(const SegmentationSystem&) = delete;
    SegmentationSystem(SegmentationSystem&&) = delete;
    SegmentationSystem& operator=(const SegmentationSystem&) = delete;
    SegmentationSystem& operator=(SegmentationSystem&&) = delete;
    ~SegmentationSystem() = default;
public:
    // NOTE: Config::voxel_size is scaled by this constant for unified parameter passing between segmentation and odometry
    static constexpr float VOXEL_SIZE_SCALAR{0.5f};
    static constexpr float VOXEL_SIZE_SCALAR_INFERENCE{0.25f};
    // scalar for map distance
    // NOTE: the idea is that we want the map to be larger than any incoming scan's max distance
    static constexpr float MAX_DISTANCE_SCALAR{1.f};
protected:
    // system configuration
    Config config_;    
};