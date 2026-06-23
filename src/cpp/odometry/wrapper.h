#pragma once

// sophus
#include <sophus/se3.hpp>
// package
#include "frame.h"
// util
#include "util/cloud.h"

// represents the semantics of a single point
using Semantics = std::pair<uint8_t, float>;

// forward declaration
struct Wrapper;

// voxel belief
struct Belief: public Voxel<Wrapper, Semantics> {
    Belief();
    virtual void update(const Semantics& data) noexcept override final;
    virtual double weight(const Semantics& data) const noexcept override final;
    virtual void finish() noexcept override final;
public:
    static constexpr size_t CLASS_COUNT{5};
    // IGNORE = 0, GROUND = 1, SHRUB = 2, STEM = 3, CANOPY = 4
    std::array<float, CLASS_COUNT> belief; 
    std::array<float, CLASS_COUNT> distribution;
};

struct Wrapper : public Frame<Wrapper, Semantics, Belief> {
    friend Belief;
    Wrapper(util::CloudView& points);
    virtual inline size_t size() const noexcept override final { return points_.size(); };
    virtual std::pair<Eigen::Vector3f&, Semantics> operator[](size_t i) noexcept override final;
    virtual std::pair<const Eigen::Vector3f&, Semantics> operator[](size_t i) const noexcept override final;
    void transform(const Sophus::SE3f& t) const noexcept;
private:
    std::span<Eigen::Vector3f> points_;
    std::span<const uint8_t> class_;
    std::span<const float> confidence_;
};