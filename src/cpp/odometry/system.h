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
#include "odometry/semantic_cloud.h"
#include "odometry/ground_plane.h"
#include "odometry/voxel_map.h"
#include "odometry/threshold.h"
// util
#include "util/cloud.h"
#include "util/profiler.h"

class OdometrySystem: util::Profiler {
public:
    struct Config {
        float max_distance{50.f};
        float max_distance_ground{10.f};
        float voxel_size{1.f};
        size_t voxel_capacity{20};
        struct {
            float initial{2.f};
            float motion{0.1f};
        } threshold;
        bool deskew{false};
        bool fixed_dt{false};
        size_t max_iterations{500};
        size_t max_threads{0};
        float convergence_criterion{0.0001f};
        struct {
            bool weighting{true};
            size_t target{0};
        } correspondence;
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
    const Sophus::SE3f& register_to_map(util::Cloud& points) noexcept; // NOTE: sets fixed_dt to true
    const Sophus::SE3f& register_to_map(util::Cloud& points, unsigned int seconds, unsigned int nanoseconds) noexcept;
public:
    struct Statistics {
        size_t iterations{0};
        float correspondences{0.f};
        struct {
            float min{0.f};
            float max{0.f};
        } threshold;
        struct {
            float linear{0.f};
            float angular{0.f};
        } velocity;
    };
    Statistics statistics;
private:
    // get the predicted delta (and handle statistics)
    std::pair<Sophus::SE3f, double> predict_delta(unsigned int seconds, unsigned int nanoseconds) noexcept;
private:
    // find correspondences
    using Correspondence = std::tuple<Eigen::Vector3f, Semantics, VoxelMap<3, SemanticCloud>::Neighbor>;
    using CorrespondenceSet = tbb::concurrent_vector<Correspondence>;
    CorrespondenceSet correspond(const SemanticCloud& points, float correspondence_distance) const noexcept;
    // constructs and solves the linear system, returning dx and chi
    std::pair<Eigen::Matrix<float, 6, 1>, float> solve(const SemanticCloud& points, const CorrespondenceSet& correspondences) noexcept;
    // returns pose and deviation respectively
    std::pair<Sophus::SE3f, Sophus::SE3f> align(SemanticCloud& points, const Sophus::SE3f& delta, Profiler& profiler);
private:
    Config config_;
    // simplified from the robust kernel, which had no appreciable benefit
    static constexpr float CORRESPONDENCE_MULTIPLIER{3.f};
    // downsampling scalar for keypoints
    static constexpr float VOXEL_SIZE_SCALAR{1.5f};
    // previous frame state
    Sophus::SE3f last_pose_;
    Sophus::SE3f last_delta_;
    unsigned int init_seconds_{std::numeric_limits<unsigned int>::max()};
    double last_stamp_{0.0};
    double prev_stamp_{0.0};
    // voxel map
    VoxelMap<3, SemanticCloud> map_;
    // adaptive threshold for correspondence search
    AdaptiveThreshold threshold_;
public:
    // EXPERIMENTAL STATE
    struct Experimental {
        GroundPlane<SemanticCloud> ground;
        VoxelMap<2> bev_prev;
        VoxelMap<2> bev_curr;
        Experimental(float max_distance, float voxel_size, size_t voxel_capacity) :
            bev_prev(max_distance, voxel_size * 2.f, voxel_capacity * 2),
            bev_curr(max_distance, voxel_size * 2.f, voxel_capacity * 2) { /* STUB */ }
    };
    Experimental experimental;
};
