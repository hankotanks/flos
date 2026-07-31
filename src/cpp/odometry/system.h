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
#include "classes.h"
// util
#include "util/cloud.h"
#include "util/profiler.h"

class OdometrySystem: util::Profiler {
public:
    struct Config { // TODO: flatten this config structure to mirror SegmentationSystem::Config
        float max_distance{50.f};
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
            bool weighting_by_belief{true}; // when false, association weights are dependent on the matched point rather than the voxel's accumulated belief
            size_t target{0};
            std::map<size_t, float> threshold; // define thresholds for target semantic classes and filter out correspondences with weights lower than...
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
    struct Belief;
    using Weight = float(*)(const Belief&, const Semantics&, const Semantics&);
    // forward declaration
    struct Wrapper;
    // voxel belief
    struct Belief: public VoxelData<Wrapper, Semantics, Weight> {
        Belief(const Weight&);
        virtual void update(const Semantics& data) noexcept override final;
        virtual float weight(const Semantics& closest, const Semantics& query) const noexcept override final { return weight_(*this, closest, query); }
        virtual void finish() noexcept override final;
    public:
        size_t prediction;
        std::array<float, classes::COUNT> belief; 
        std::array<float, classes::COUNT> confidence;
    private:
        Weight weight_;
    };
    // util::Cloud wrapper for fast access of fields
    struct Wrapper : public Frame<Wrapper, Semantics, Belief, Weight> {
        friend Belief;
        Wrapper(util::Cloud& points);
        virtual inline size_t size() const noexcept override final { return points_.size(); };
        void transform(const Sophus::SE3f& t) const noexcept;
        virtual std::pair<Eigen::Vector3f&, Semantics> operator[](size_t i) noexcept override final;
        virtual std::pair<const Eigen::Vector3f&, Semantics> operator[](size_t i) const noexcept override final;
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
    Sophus::SE3f prologue(unsigned int seconds, unsigned int nanoseconds) noexcept;
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
    static constexpr float VOXEL_SIZE_SCALAR{1.f}; // TODO: remove when I'm confident it won't change anymore
    // simplified from the robust kernel, which had no appreciable benefit
    static constexpr float CORRESPONDENCE_MULTIPLIER{3.f};
    // lookup table for config_.correspondence.threshold
    std::vector<float> correspondence_threshold_;
    // previous frame state
    Sophus::SE3f last_pose_;
    Sophus::SE3f last_delta_;
    unsigned int init_seconds_{std::numeric_limits<unsigned int>::max()};
    float last_stamp_{0.f};
    float prev_stamp_{0.f};
    // voxel map
    VoxelMap<Wrapper> map_;
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
