#pragma once

// sophus
#include <sophus/se3.hpp>
// tbb
#include <tbb/global_control.h>
#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>
// package
#include "voxel_map.h"
// util
#include "util/cloud.h"

class OdometrySystem {
public:
    struct Config {
        struct {
            double max_distance{100.0};
            double voxel_size{1.0}; // NOTE: downsampling approach has changed from kiss-icp
            size_t voxel_capacity{20};
        } map;
        struct {
            double threshold_initial{2.0};
            double min_motion{0.1};
        } threshold;
        struct {
            size_t max_iterations{500};
            size_t max_threads{0};
            double convergence_criterion{0.0001};
            double correspondence_multiplier{3.0};
        } registration;
    };
public:
    explicit OdometrySystem(const Config& config) : config_(config),
        map_(config.map.max_distance, config.map.voxel_size, config.map.voxel_capacity),
        threshold_(config.map.max_distance, config.threshold.threshold_initial, config.threshold.min_motion) {
        size_t max_threads = config.registration.max_threads > 0 ? \
            config.registration.max_threads : \
            tbb::this_task_arena::max_concurrency();
        volatile static const tbb::global_control _ = tbb::global_control( // NOTE: static storage type is required here
            tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_threads));
    }
public:
    OdometrySystem() = delete;
    OdometrySystem(const OdometrySystem&) = delete;
    OdometrySystem(OdometrySystem&&) = delete;
    OdometrySystem& operator=(const OdometrySystem&) = delete;
    OdometrySystem& operator=(OdometrySystem&&) = delete;
    ~OdometrySystem() = default;
public:
    // main registration function
    const Sophus::SE3f& register_to_map(util::CloudView&& points) noexcept;
    // returns the current local map
    inline const util::CloudView& cloud() const noexcept { return map_.cloud(); }
private:
    class AdaptiveThreshold {
    public:
        explicit AdaptiveThreshold(double max_distance, double initial, double min_motion) : 
            max_distance_(max_distance), model_sse_(initial * initial), min_motion_(min_motion) { /* STUB */ }
        AdaptiveThreshold() = delete;
        AdaptiveThreshold(const AdaptiveThreshold&) = delete;
        AdaptiveThreshold(AdaptiveThreshold&&) = delete;
        AdaptiveThreshold& operator=(const AdaptiveThreshold&) = delete;
        AdaptiveThreshold& operator=(AdaptiveThreshold&&) = delete;
        ~AdaptiveThreshold() = default;
    public:
        inline double compute() const noexcept { return std::sqrt(model_sse_ / static_cast<double>(num_sample_)); }
        void update(const Sophus::SE3f& deviation) noexcept;
    private:
        double max_distance_;
        double model_sse_;
        double min_motion_;
        size_t num_sample_{1};
    };
private:
    // constructs and solves the linear system, returning dx
    Eigen::Matrix<float, 6, 1> solve(const util::CloudView& points, double sigma) const noexcept;
    // returns pose and deviation respectively
    std::pair<Sophus::SE3f, Sophus::SE3f> align(const util::CloudView& points) const;
private:
    Config config_;
    Sophus::SE3f last_pose_;
    Sophus::SE3f last_delta_;
    VoxelMap map_;
    AdaptiveThreshold threshold_;
};