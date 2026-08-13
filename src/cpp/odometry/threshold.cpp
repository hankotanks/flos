#include "odometry/threshold.h"

void AdaptiveThreshold::update(const Sophus::SE3f& deviation) noexcept {
    const float model_error = std::invoke([this, &deviation]() {
        const float theta = Eigen::AngleAxisf(deviation.rotationMatrix()).angle();
        const float delta_rot = 2.0 * this->max_distance_ * std::sin(theta / 2.0);
        const float delta_trans = deviation.translation().norm();
        return delta_trans + delta_rot;
    });
    if(model_error > threshold_motion_) {
        sse_ += model_error * model_error;
        samples_++;
    }
}