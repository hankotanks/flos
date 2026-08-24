#pragma once

// system
#include <filesystem>
#include <set>
// pybind
#include <pybind11/embed.h>
// util
#include "util/cloud.h"
#include "util/profiler.h"

class SegmentationSystem : util::Profiler {
    friend class OfflineNode;
public:
    struct Config {
        float max_distance{50.f};
        float voxel_size{1.f};
        float voxel_size_inference{0.1f};
        bool double_downsample{false};
        struct {
            std::set<size_t> ignore{0};
            std::map<size_t, size_t> remap;
        } semantics;
    public:
        friend std::ostream& operator<<(std::ostream& stream, const Config& config);
    };
public:
    class Model {
        friend SegmentationSystem;
        explicit Model(const SegmentationSystem::Config& config, const std::filesystem::path& path_to_model, util::Profiler* profiler);
        Model() = delete;
        Model(const Model&) = delete;
        Model(Model&&) = delete;
        Model& operator=(const Model&) = delete;
        Model& operator=(Model&&) = delete;
    public:
        ~Model() = default;
    public:
        void infer(util::Cloud& cloud);
        void preprocess(util::Cloud& cloud);
    private:
        util::Profiler* profiler_;
        py::object module_;
        py::object helper_;
        py::object infer_;
        const SegmentationSystem::Config& config_;
    };
public:
    explicit SegmentationSystem(const Config& config) : util::Profiler("SegmentationSystem"), config_(config) { /* STUB */ }
    SegmentationSystem() = delete;
    SegmentationSystem(const SegmentationSystem&) = delete;
    SegmentationSystem(SegmentationSystem&&) = delete;
    SegmentationSystem& operator=(const SegmentationSystem&) = delete;
    SegmentationSystem& operator=(SegmentationSystem&&) = delete;
    ~SegmentationSystem() = default;
public:
    void preprocess(util::Cloud& cloud);
    SegmentationSystem::Model load(const std::filesystem::path& path_to_model);
private:
    // NOTE: Config::voxel_size is scaled by this constant for unified parameter passing between segmentation and odometry
    static constexpr float VOXEL_SIZE_SCALAR{0.5f};
    // system configuration
    Config config_;    
};