#include "wrapper.h"

Belief::Belief() { 
    // populate will an flat distribution
    belief.fill(std::log(1.f / static_cast<float>(CLASS_COUNT))); 
}

void Belief::update(const Semantics& data) noexcept {
    const auto& [point_class, point_confidence] = data;
    // TODO: implement
    double prob_hit = std::log(point_confidence);
    double prob_miss = std::log((1.0 - point_confidence) / (static_cast<double>(Belief::CLASS_COUNT) - 1.0));
    for(size_t i = 0; i < Belief::CLASS_COUNT; ++i) {
        belief[i] += (point_class == i) ? prob_hit : prob_miss;
    }
}

double Belief::weight(const Semantics& data) const noexcept {
    // TODO: implement
    return 1.0;
}

Wrapper::Wrapper(util::CloudView& points) : 
    points_(points.points()), 
    class_(points.field<uint8_t>("class")), 
    confidence_(points.field<float>("confidence")) { /* STUB */ }

std::pair<Eigen::Vector3f&, Semantics> Wrapper::operator[](size_t i) noexcept { 
    return std::pair<Eigen::Vector3f&, Semantics>(points_[i], std::make_pair(class_[i], confidence_[i])); 
}

std::pair<const Eigen::Vector3f&, Semantics> Wrapper::operator[](size_t i) const noexcept { 
    return std::pair<const Eigen::Vector3f&, Semantics>(points_[i], std::make_pair(class_[i], confidence_[i])); 
}

void Wrapper::transform(const Sophus::SE3f& t) const noexcept { 
    std::for_each(points_.begin(), points_.end(), [&t](Eigen::Vector3f& p) { p = t * p; }); 
}