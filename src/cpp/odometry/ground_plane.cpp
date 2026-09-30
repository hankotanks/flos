#include "odometry/ground_plane.h"

GroundPlane::GroundPlane(float max_distance, float inlier_ratio, float inlier_distance) : max_distance_(max_distance), inlier_distance_(inlier_distance) {
    float p = std::pow(inlier_ratio, 3);
    iterations_ = (p >= 1.f) ? 1 : static_cast<size_t>(std::ceil(std::log(1.f - GroundPlane::CONFIDENCE) / std::log(1.f - p)));
}

Sophus::SE3f GroundPlane::lift(const Sophus::SE2f& T) const noexcept {
    Eigen::Matrix3f R_plane = Eigen::Matrix3f::Identity();
    R_plane.block<2, 2>(1, 1) = T.so2().matrix();
    const Eigen::Matrix3f R = plane_ * R_plane * plane_.transpose();
    const Eigen::Vector3f t = centroid_ + plane_.block<3, 2>(0, 1) * T.translation() - R * centroid_;
    return Sophus::SE3f{R, t};
}

std::function<void(Eigen::Vector2f&, float&)>
GroundPlane::project(const Sophus::SE3f& T) const noexcept {
    const Eigen::Matrix3f& R = T.so3().matrix();
    const Eigen::Vector3f& t = T.translation();

    Eigen::Matrix<float, 3, 2> basis;
    basis.col(0) = t1_;
    basis.col(1) = t2_;

    const Eigen::Matrix2f A = basis.transpose() * R * basis;
    const Eigen::Vector2f b = basis.transpose() * (R * centroid_ + t - centroid_);
    const Eigen::Vector2f s = basis.transpose() * R * n_;
    const Eigen::RowVector2f dx = n_.transpose() * R * basis;

    const float ds = n_.dot(R * n_);
    const float db = n_.dot(R * centroid_ + t - centroid_);

    return [A, b, s, dx, ds, db](Eigen::Vector2f& point, float& distance) {
        const Eigen::Vector2f point_prev = point;
        const float distance_prev = distance;
        point = A * point_prev + b + distance_prev * s;
        distance = dx.dot(point_prev) + ds * distance_prev + db;
    };
}

Eigen::Vector3f GroundPlane::project(const Eigen::Vector3f& point) const noexcept {
    Eigen::Vector3f local = point - centroid_;
    return {t1_.dot(local), t2_.dot(local), local.dot(n_)};
}

std::pair<Eigen::Vector2f, float> BirdsEyeCloud::operator[](size_t i) const noexcept {
    const Eigen::Vector3f& point = points_[i];
    return std::pair<Eigen::Vector2f, float>(std::piecewise_construct, std::forward_as_tuple(point.x(), point.y()), std::forward_as_tuple(point.z()));
}