#pragma once

// sophus
#include <sophus/se3.hpp>

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