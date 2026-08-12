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
#include "odometry/bev.h"
#include "odometry/voxel_map.h"
#include "classes.h"
// util
#include "util/cloud.h"
#include "util/profiler.h"

class OdometrySystem: util::Profiler {
public:
    struct Config {
        float max_distance{50.f};
        float max_distance_fit{10.f}; // TODO: add full parameter support
        float voxel_size{1.f}; // NOTE: downsampling approach has changed from kiss-icp
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
            bool weighting{true}; // weights associations based on the aggreement between the point's class distribution and the voxel's accumulated belief
            // NOTE: a current limitation is that when weighting_by_belief is false, the voxel belief is still used for the weight
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
    // represents the semantics of a single point
    struct Semantics {
        size_t prediction;
        std::array<float, classes::COUNT> confidence;
        explicit Semantics() = default;
    };
    // forward declaration
    struct Wrapper;
    // voxel belief
    struct Belief: public VoxelData<Wrapper, Semantics> {
        explicit Belief();
        virtual void update(const Semantics& data) noexcept override final;
        virtual void finish() noexcept override final;
        virtual bool accept(const Semantics& data) const noexcept override final;
        virtual float weight(const Semantics& query) const noexcept override final;
    public:
        size_t prediction;
        std::array<float, classes::COUNT> belief; 
        std::array<float, classes::COUNT> confidence;
    };
    // util::Cloud wrapper for fast access of fields
    struct Wrapper : public Frame<Wrapper, Semantics, Belief> {
        friend Belief;
        Wrapper(util::Cloud& points);
        virtual inline size_t size() const noexcept override final { return points_.size(); };
        void transform(const Sophus::SE3f& t) const noexcept;
        virtual std::pair<Eigen::Vector3f&, Semantics> operator[](size_t i) noexcept override final;
        virtual std::pair<const Eigen::Vector3f&, Semantics> operator[](size_t i) const noexcept override final;
    public:
        inline BEV project(const Eigen::Vector3f& centroid, const Eigen::Matrix3f& ground, float voxel_size, size_t voxel_capacity) { return BEV{centroid, ground, points_, class_, voxel_size, voxel_capacity}; }
    private:
        Semantics data(size_t i) const noexcept;
    private:
        std::span<Eigen::Vector3f> points_;
        std::span<const uint8_t> class_;
        std::array<std::span<const float>, classes::COUNT> confidence_;
    };
public:
    // returns the current local map
    // NOTE: this is slow
    const VoxelMap<Wrapper>::Cloud& cloud() const noexcept { return map_.cloud(); }
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
    using Correspondence = std::tuple<Eigen::Vector3f, Semantics, VoxelMap<Wrapper>::Neighbor>;
    using CorrespondenceSet = tbb::concurrent_vector<Correspondence>;
    CorrespondenceSet correspond(const Wrapper& points, float correspondence_distance) const noexcept;
    // constructs and solves the linear system, returning dx and chi
    std::pair<Eigen::Matrix<float, 6, 1>, float> solve(const Wrapper& points, const CorrespondenceSet& correspondences) noexcept;
    // returns pose and deviation respectively
    std::pair<Sophus::SE3f, Sophus::SE3f> align(Wrapper& points, const Sophus::SE3f& delta, Profiler& profiler);
private:
    Config config_;
    // simplified from the robust kernel, which had no appreciable benefit
    static constexpr float CORRESPONDENCE_MULTIPLIER{3.f};
    // previous frame state
    Sophus::SE3f last_pose_;
    Sophus::SE3f last_delta_;
    unsigned int init_seconds_{std::numeric_limits<unsigned int>::max()};
    double last_stamp_{0.0};
    double prev_stamp_{0.0};
    // voxel map
    VoxelMap<Wrapper> map_;
public:
    // EXPERIMENTAL STATE
    struct Experimental {
        Eigen::Vector3f centroid;
        Eigen::Matrix3f plane;
        BEV bev_prev;
        BEV bev_curr;
        BEV bev_curr_post;
        BEV bev_prev_next;
        Sophus::SE3f pose_initial;
        Sophus::SE3f pose_refined;
        Sophus::SE2f T_bev_projected;
        Sophus::SE3f T_bev;
        std::vector<Eigen::Vector3f> stem_initial;
        std::vector<Eigen::Vector3f> stem_refined;
        bool accepted{true};
        float correspondence_distance{0.f};
    };
    Experimental experimental;
    std::pair<Eigen::Vector3f, Eigen::Matrix3f> fit_ground_plane(const Eigen::Vector3f& origin) noexcept;
    Sophus::SE3f lift(const Sophus::SE2f& se2, const Eigen::Vector3f& centroid, const Eigen::Matrix3f& plane);
private:
    class AdaptiveThreshold {
    public:
        explicit AdaptiveThreshold(float max_distance, float threshold_initial, float threshold_motion) : 
            max_distance_(max_distance), sse_(threshold_initial * threshold_initial), threshold_motion_(threshold_motion) { /* STUB */ }
        AdaptiveThreshold() = delete;
        AdaptiveThreshold(const AdaptiveThreshold&) = delete;
        AdaptiveThreshold(AdaptiveThreshold&&) = delete;
        AdaptiveThreshold& operator=(const AdaptiveThreshold&) = delete;
        AdaptiveThreshold& operator=(AdaptiveThreshold&&) = delete;
        ~AdaptiveThreshold() = default;
    public:
        inline float compute() const noexcept { return std::sqrt(sse_ / static_cast<float>(samples_)); }
        void update(const Sophus::SE3f& deviation) noexcept;
    private:
        float max_distance_;
        float sse_;
        float threshold_motion_;
        size_t samples_{1};
    };
    // adaptive threshold member
    AdaptiveThreshold threshold_;
};
