#include "wrapper.h"

Belief::Belief() { 
    // populate will an flat distribution
    belief.fill(std::log(1.f / static_cast<float>(Semantics::CLASS_COUNT)));
    // NOTE: distribution will be populated by the VoxelMap::update, so we don't need to worry about it
}

void Belief::update(const Semantics& data) noexcept {
    static constexpr float EPS = 1e-6f;
    for(size_t i = 0; i < Semantics::CLASS_COUNT; ++i) {
        belief[i] += std::log(std::max(data.confidence[i], EPS));
    }
}

// TODO: This is NOT final
float Belief::weight(const Semantics& data) const noexcept {
    float w = 0.0;
    // NOTE: currently, dot product
    // consider KL or Jensen-Shannon
    for(size_t i = 0; i < Semantics::CLASS_COUNT; ++i) w += confidence[i] * data.confidence[i];
    return w;
}

void Belief::finish() noexcept {
    float maxima = this->belief[0];
    for(size_t i = 1; i < Semantics::CLASS_COUNT; ++i) maxima = std::max(maxima, belief[i]);
    for(size_t i = 0; i < Semantics::CLASS_COUNT; ++i) confidence[i] = std::exp(belief[i] - maxima);
    float sum = confidence[0];
    for(size_t i = 1; i < Semantics::CLASS_COUNT; ++i) sum += confidence[i];
    for(size_t i = 0; i < Semantics::CLASS_COUNT; ++i) confidence[i] /= sum;
}

Wrapper::Wrapper(util::CloudView& points) : points_(points.points()), class_(points.field<uint8_t>("class")) {
    for(size_t i = 0; i < Semantics::CLASS_COUNT; ++i) {
        confidence_[i] = points.field_opt<float>(std::to_string(i));
    }
}

Semantics Wrapper::data(size_t i) const noexcept {
    Semantics data;
    data.prediction = class_[i];
    for(size_t j = 0; j < Semantics::CLASS_COUNT; ++j) {
        auto& field = confidence_[j];
        data.confidence[j] = field ? (*field)[i] : 0.f;
    }
    return data;
}

void Wrapper::transform(const Sophus::SE3f& t) const noexcept { 
    std::for_each(points_.begin(), points_.end(), [&t](Eigen::Vector3f& p) { p = t * p; }); 
}

std::pair<Eigen::Vector3f&, Semantics> Wrapper::operator[](size_t i) noexcept { 
    return std::pair<Eigen::Vector3f&, Semantics>(points_[i], Wrapper::data(i)); 
}

std::pair<const Eigen::Vector3f&, Semantics> Wrapper::operator[](size_t i) const noexcept { 
    return std::pair<const Eigen::Vector3f&, Semantics>(points_[i], Wrapper::data(i)); 
}