#include "wrapper.h"

Belief::Belief() { 
    // populate will an flat distribution
    belief.fill(std::log(1.f / static_cast<float>(CLASS_COUNT)));
    // NOTE: distribution will be populated by the VoxelMap::update, so we don't need to worry about it
}

// TODO: This is NOT final
void Belief::update(const Semantics& data) noexcept {
    const auto& [point_class, point_confidence] = data;
    // TODO: implement
    double prob_hit = std::log(point_confidence);
    double prob_miss = std::log((1.0 - point_confidence) / (static_cast<double>(Belief::CLASS_COUNT) - 1.0));
    for(size_t i = 0; i < Belief::CLASS_COUNT; ++i) {
        belief[i] += (point_class == i) ? prob_hit : prob_miss;
    }
}

// TODO: This is NOT final
double Belief::weight(const Semantics& data) const noexcept {
    // compute the distribution of the query
    std::array<float, CLASS_COUNT> query = [&data]() {
        std::array<float, CLASS_COUNT> b;
        const auto& [point_class, point_confidence] = data;
        float other_confidence = (1.f - point_confidence) / (static_cast<float>(CLASS_COUNT) - 1.0);
        for(size_t i = 0; i < CLASS_COUNT; ++i) {
            b[i] = (i == point_class) ? point_confidence : other_confidence;
            b[i] = std::max(b[i], 1e-6f);
        }
        return std::move(b);
    }();
    double w = 0.0;
    // NOTE: currently, dot product
    // consider KL or Jensen-Shannon
    for(size_t i = 0; i < CLASS_COUNT; ++i) w += query[i] * distribution[i];
    return w;
}

void Belief::finish() noexcept {
    float maxima = this->belief[0];
    for(size_t i = 1; i < CLASS_COUNT; ++i) maxima = std::max(maxima, belief[i]);
    for(size_t i = 0; i < CLASS_COUNT; ++i) distribution[i] = std::exp(belief[i] - maxima);
    float sum = distribution[0];
    for(size_t i = 1; i < CLASS_COUNT; ++i) sum += distribution[i];
    for(size_t i = 0; i < CLASS_COUNT; ++i) distribution[i] /= sum;
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