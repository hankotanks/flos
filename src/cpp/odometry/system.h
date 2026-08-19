#pragma once

// system
#include <limits>
#include <set>
// sophus
#include <sophus/se3.hpp>
// tbb
#include <tbb/global_control.h>
#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>
// package
#include "odometry/adaptive_threshold.h"
#include "odometry/semantic_cloud.h"
#include "odometry/projection.h"
#include "odometry/voxel_map.h"
// util
#include "util/cloud.h"
#include "util/profiler.h"

class OdometrySystem: util::Profiler {
    friend class OdometryNode;
    friend class OfflineNode;
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
        bool deskew{true};
        bool constant_scan_rate{false};
        size_t max_iterations{500};
        size_t max_threads{0};
        float convergence_criterion{0.0001f};
        struct {
            bool weighting{true};
            size_t target{0};
        } correspondence;
        struct {
            bool enable{true};
            float iou{0.73f};
            std::set<size_t> ground{1};
            std::set<size_t> target{3};
        } projection;
    public:
        friend std::ostream& operator<<(std::ostream& stream, const Config& config);
    };
public:
    explicit OdometrySystem(const Config& config);
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
            float linear{0.f};
            float angular{0.f};
        } velocity;
        bool refined{false};
    };
private:
    static constexpr float CORRESPONDENCE_DISTANCE_SCALAR{3.f};
    // downsampling scalar for keypoints
    static constexpr float VOXEL_SIZE_SCALAR{1.5f};
    // configuration
    Config config_;
    // statistics
    Statistics statistics_;
    // frame state
    size_t frame_{0};
    unsigned int init_seconds_{std::numeric_limits<unsigned int>::max()};
    double last_stamp_{0.0};
    double prev_stamp_{0.0};
    Sophus::SE3f last_pose_;
    Sophus::SE3f last_delta_;
    // voxel map
    VoxelMap<3, SemanticCloud> map_;
    // current and previous frame projections
    Projection projection_;
    AdaptiveThreshold threshold_;
private:
    // get the predicted delta (and handle statistics)
    std::pair<Sophus::SE3f, double> predict(unsigned int seconds, unsigned int nanoseconds) noexcept;
private:
    // find correspondences
    using Correspondence = std::pair<Eigen::Vector3f, VoxelMap<3, SemanticCloud>::Adjacency>;
    using CorrespondenceSet = tbb::concurrent_vector<Correspondence>;
    CorrespondenceSet correspond(const SemanticCloud& points, float correspondence_distance) const noexcept;
    // constructs and solves the linear system, returning dx and chi
    std::pair<Eigen::Matrix<float, 6, 1>, float> solve(const CorrespondenceSet& correspondences) noexcept;
private:
    // assumes points are already transformed by initial_guess
    // updates initial guess (and points)
    Sophus::SE3f refine(SemanticCloud& points, Sophus::SE3f& initial_guess, float correspondence_distance);
    // returns pose and deviation respectively
    Sophus::SE3f align(SemanticCloud& points, const Sophus::SE3f& initial_guess, float correspondence_distance);
};
