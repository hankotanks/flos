#pragma once

// sophus
#include <sophus/se3.hpp>
// tbb
#include <tbb/global_control.h>
#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>
// package
#include "voxel_map.h"
#include "wrapper.h"
// util
#include "util/cloud.h"

class OdometrySystem {
public:
    struct Config {
        struct {
            double max_distance{25.0};
            double voxel_size{1.0}; // NOTE: downsampling approach has changed from kiss-icp
            size_t voxel_capacity{20};
        } map;
        struct {
            double threshold_initial{2.0};
            double min_motion{0.1};
        } threshold;
        struct {
            bool confidence_weighting{true};
            bool correspondence_restriction{true};
            size_t max_iterations{500};
            size_t max_threads{0};
            double convergence_criterion{0.0001};
            double correspondence_multiplier{3.0};
        } registration;
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
    using Semantics = std::pair<uint8_t, float>;
    // main registration function
    const Sophus::SE3f& register_to_map(util::CloudView&& points) noexcept;
    // returns the current local map
    const VoxelMap<Wrapper>::Cloud& cloud() const noexcept { return map_.cloud(); }
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
    Eigen::Matrix<float, 6, 1> solve(const Wrapper& points, double sigma) const noexcept;
    // returns pose and deviation respectively
    std::pair<Sophus::SE3f, Sophus::SE3f> align(Wrapper& points) const;
private:
    Config config_;
    // NOTE: see note in segmentation/node.h
    static constexpr double VOXEL_SIZE_SCALAR{1.5};
    Sophus::SE3f last_pose_;
    Sophus::SE3f last_delta_;
    VoxelMap<Wrapper> map_;
    AdaptiveThreshold threshold_;
};