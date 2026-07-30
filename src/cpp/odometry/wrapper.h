#pragma once

// sophus
#include <sophus/se3.hpp>
// package
#include "odometry/voxel_map_frame.h"
#include "classes.h"
// util
#include "util/cloud.h"

// represents the semantics of a single point
struct Semantics {
    size_t prediction;
    std::array<float, classes::COUNT> confidence;
    explicit Semantics() = default;
};

// forward declaration
struct Wrapper;

// voxel belief
struct Belief: public Voxel<Wrapper, Semantics> {
    Belief();
    virtual void update(const Semantics& data) noexcept override final;
    virtual float weight(const Semantics& data) const noexcept override final;
    virtual void finish() noexcept override final;
public:
    size_t prediction;
    std::array<float, classes::COUNT> belief; 
    std::array<float, classes::COUNT> confidence;
};

struct Wrapper : public Frame<Wrapper, Semantics, Belief> {
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