#pragma once

// package
#include "classes.h"
#include "odometry/voxel_map.h"
// util
#include "util/cloud.h"

// represents the semantics of a single point
struct Semantics {
    size_t prediction;
    std::array<float, classes::COUNT> confidence;
    explicit Semantics() = default;
};

// forward declaration
struct SemanticCloud;

// voxel belief
struct Belief: public VoxelDataBase<SemanticCloud, Semantics> {
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
struct SemanticCloud : public Frame<SemanticCloud, 3, Semantics, Belief> {
    friend Belief;
    SemanticCloud(util::Cloud& points);
    virtual inline size_t size() const noexcept override final { return points_.size(); };
    void transform(const Sophus::SE3f& t) const noexcept;
    virtual std::pair<Eigen::Vector3f, Semantics> operator[](size_t i) const noexcept override final;
private:
    std::span<Eigen::Vector3f> points_;
    std::span<const uint8_t> class_;
    std::array<std::span<const float>, classes::COUNT> confidence_;
};