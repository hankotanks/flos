#pragma once

// system
#include <limits>
// sophus
#include <sophus/se3.hpp>
// tbb
#include <tbb/global_control.h>
#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>
// package
#include "odometry/voxel_map.h"
#include "odometry/wrapper.h"
// util
#include "util/cloud.h"
#include "util/profiler.h"

class OdometrySystem: util::Profiler {
public:
    struct Config { // TODO: flatten this config structure to mirror SegmentationSystem::Config
        struct {
            float max_distance{50.f};
            std::map<size_t, float> min_weight;
            float voxel_size{1.f}; // NOTE: downsampling approach has changed from kiss-icp
            size_t voxel_capacity{20};
        } map;
        struct {
            float threshold_initial{2.f};
            float min_motion{0.1f};
        } threshold;
        struct {
            bool confidence_weighting{true};
            bool correspondence_restriction{true};
            bool correspondence_with_voxel_belief{true}; // TODO: this is an outdated parameter, since the final system will only use voxel belief. keep for now for future ablation study
            bool constant_velocity{true};
            size_t max_iterations{500};
            size_t max_threads{0};
            float convergence_criterion{0.0001f};
            float correspondence_multiplier{3.f};
        } registration;
    public:
        friend std::ostream& operator<<(std::ostream& stream, const Config& config);
    };
public:
    explicit OdometrySystem(const Config& config);
public:
    OdometrySystem() = delete;
    OdometrySystem(const OdometrySystem&) = delete;
    OdometrySystem(OdometrySystem&&) = delete;
    OdometrySystem& operator=(const OdometrySystem&) = delete;
    OdometrySystem& operator=(OdometrySystem&&) = delete;
    ~OdometrySystem() = default;
public:
    // main registration function
    // NOTE: deskews and transforms points by the solution pose
    const Sophus::SE3f& register_to_map(util::Cloud& points, unsigned int seconds, unsigned int nanoseconds) noexcept;
    // returns the current local map
    const VoxelMap<Wrapper>::Cloud& cloud() const noexcept { return map_.cloud(); }
public:
    struct Statistics {
        size_t iterations{0};
        struct {
            size_t initial_count{0};
            size_t final_count{0};
            float mean{0.f};
        } correspondence;
        struct {
            struct {
                float initial{0.f};
                float solved{0.f};
            } linear;
            struct {
                float initial{0.f};
                float solved{0.f};
            } angular;
        } velocity;
    };
    Statistics statistics;
private:
    class AdaptiveThreshold {
    public:
        explicit AdaptiveThreshold(float max_distance, float initial, float min_motion) : 
            max_distance_(max_distance), model_sse_(initial * initial), min_motion_(min_motion) { /* STUB */ }
        AdaptiveThreshold() = delete;
        AdaptiveThreshold(const AdaptiveThreshold&) = delete;
        AdaptiveThreshold(AdaptiveThreshold&&) = delete;
        AdaptiveThreshold& operator=(const AdaptiveThreshold&) = delete;
        AdaptiveThreshold& operator=(AdaptiveThreshold&&) = delete;
        ~AdaptiveThreshold() = default;
    public:
        inline float compute() const noexcept { return std::sqrt(model_sse_ / static_cast<float>(num_sample_)); }
        void update(const Sophus::SE3f& deviation) noexcept;
    private:
        float max_distance_;
        float model_sse_;
        float min_motion_;
        size_t num_sample_{1};
    };
private:
    // get the predicted delta (and handle statistics)
    void prologue(Sophus::SE3f& delta, unsigned int seconds, unsigned int nanoseconds) noexcept;
    // handle end of frame (saving last_pose and final velocity statistics)
    void epilogue(const Sophus::SE3f& pose, unsigned int seconds, unsigned int nanoseconds) noexcept;
private:
    // find correspondences
    using Correspondence = std::tuple<Eigen::Vector3f, Semantics, VoxelMap<Wrapper>::Neighbor>;
    using CorrespondenceSet = tbb::concurrent_vector<Correspondence>;
    CorrespondenceSet correspond(const Wrapper& points, float sigma) const noexcept;
    // constructs and solves the linear system, returning dx
    Eigen::Matrix<float, 6, 1> solve(const Wrapper& points, const CorrespondenceSet& correspondences) noexcept;
    // returns pose and deviation respectively
    std::pair<Sophus::SE3f, Sophus::SE3f> align(Wrapper& points, const Sophus::SE3f& delta);
private:
    Config config_;
    // NOTE: see note in segmentation/node.h
    static constexpr float VOXEL_SIZE_SCALAR{1.f};
    Sophus::SE3f last_pose_;
    Sophus::SE3f last_delta_;
    unsigned int init_seconds_{std::numeric_limits<unsigned int>::max()};
    float last_stamp_{0.f};
    float prev_stamp_{0.f};
    using Predicate = bool(*)(const Belief& data, Semantics point, Semantics query);
    Predicate pred_;
    VoxelMap<Wrapper> map_;
    AdaptiveThreshold threshold_;
};