#include "odometry/semantic_cloud.h"

Belief::Belief() {
    // populate will a flat distribution
    // NOTE: distribution will be populated by the VoxelMap::update, so we don't need to worry about it
    static constexpr float UNIFORM_CONFIDENCE = 1.f / static_cast<float>(classes::COUNT);
    confidence.fill(UNIFORM_CONFIDENCE);
    static const float UNIFORM_BELIEF = std::log(UNIFORM_CONFIDENCE);
    belief.fill(UNIFORM_BELIEF);
    prediction = std::numeric_limits<size_t>::max();
}

void Belief::update(const Semantics& data) noexcept {
    static constexpr float EPS = 1e-6f;
    for(size_t i = 0; i < classes::COUNT; ++i) {
        belief[i] += std::log(std::max(data.confidence[i], EPS));
    }
}

void Belief::finish() noexcept {
    auto it = std::max_element(belief.begin(), belief.end());
    // update predicted class of this voxel
    prediction = std::distance(belief.begin(), it);
    // renormalize belief so it doesn't keep growing
    float maxima = *it;
    for(auto& b : belief) b -= maxima;
    // compute confidence probabilities for future VoxelMap::closest queries
    for(size_t i = 0; i < classes::COUNT; ++i) confidence[i] = std::exp(belief[i]);
    float sum = confidence[0];
    for(size_t i = 1; i < classes::COUNT; ++i) sum += confidence[i];
    for(size_t i = 0; i < classes::COUNT; ++i) confidence[i] /= sum;
}

bool Belief::accept(const Semantics& query) const noexcept {
    return prediction == query.prediction;
}

#if 0
float Belief::weight(const Semantics& query) const noexcept {
    float jsd = 0.f;
    for(size_t i = 0; i < classes::COUNT; ++i) {
        const float p = confidence[i];
        const float q = query.confidence[i];
        const float m = 0.5f * (p + q);
        if(m == 0.f) continue;
        if(p > 0.f) jsd += 0.5f * p * std::log2(p / m);
        if(q > 0.f) jsd += 0.5f * q * std::log2(q / m);
    }
    return 1.f - std::sqrt(std::clamp(jsd, 0.f, 1.f));
}
#else
float Belief::weight(const Semantics& query) const noexcept {
    float w = 0.f;
    for(size_t i = 0; i < classes::COUNT; ++i) w += confidence[i] * query.confidence[i];
    assert(w >= 0.f && w <= 1.f);
    return w;
}
#endif

SemanticCloud::SemanticCloud(util::Cloud& points) : points_(points.points()), class_(points.field<uint8_t>("class")) {
    for(size_t i = 0; i < classes::COUNT; ++i) {
        confidence_[i] = points.field<float>(std::to_string(i));
    }
}

Semantics SemanticCloud::data(size_t i) const noexcept {
    Semantics data;
    data.prediction = class_[i];
    for(size_t j = 0; j < classes::COUNT; ++j) {
        data.confidence[j] = confidence_[j][i];
    }
    return data;
}

void SemanticCloud::transform(const Sophus::SE3f& t) const noexcept { 
    std::for_each(points_.begin(), points_.end(), [&t](Eigen::Vector3f& p) { p = t * p; }); 
}

std::pair<Eigen::Vector3f&, Semantics> SemanticCloud::operator[](size_t i) noexcept { 
    return std::pair<Eigen::Vector3f&, Semantics>(points_[i], SemanticCloud::data(i)); 
}

std::pair<const Eigen::Vector3f&, Semantics> SemanticCloud::operator[](size_t i) const noexcept { 
    return std::pair<const Eigen::Vector3f&, Semantics>(points_[i], SemanticCloud::data(i)); 
}
