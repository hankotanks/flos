#include "odometry/ground_plane.h"

std::pair<Eigen::Vector2f, float> BirdsEyeCloud::operator[](size_t i) const noexcept {
    const Eigen::Vector3f& point = points_[i];
    return std::pair<Eigen::Vector2f, float>(std::piecewise_construct, std::forward_as_tuple(point.x(), point.y()), std::forward_as_tuple(point.z()));
}

Sophus::SE3f GroundPlane::lift(const Sophus::SE2f& T) const noexcept {
    Eigen::Matrix3f R_plane = Eigen::Matrix3f::Identity();
    R_plane.block<2, 2>(1, 1) = T.so2().matrix();
    const Eigen::Matrix3f R = plane_ * R_plane * plane_.transpose();
    const Eigen::Vector3f t = centroid + plane_.block<3, 2>(0, 1) * T.translation() - R * centroid;
    return Sophus::SE3f{R, t};
}

std::function<void(Eigen::Vector2f&, float&)>
GroundPlane::project(const Sophus::SE3f& T) const noexcept {
    const Eigen::Matrix3f& R = T.so3().matrix();
    const Eigen::Vector3f& t = T.translation();

    Eigen::Matrix<float, 3, 2> basis;
    basis.col(0) = t1;
    basis.col(1) = t2;

    const Eigen::Matrix2f A = basis.transpose() * R * basis;
    const Eigen::Vector2f b = basis.transpose() * (R * centroid + t - centroid);
    const Eigen::Vector2f s = basis.transpose() * R * n;
    const Eigen::RowVector2f dx = n.transpose() * R * basis;

    const float ds = n.dot(R * n);
    const float db = n.dot(R * centroid + t - centroid);

    return [A, b, s, dx, ds, db](Eigen::Vector2f& point, float& distance) {
        const Eigen::Vector2f point_prev = point;
        const float distance_prev = distance;
        point = A * point_prev + b + distance_prev * s;
        distance = dx.dot(point_prev) + ds * distance_prev + db;
    };
}

std::function<void(Eigen::Vector2f&, float&)>
GroundPlane::project(const GroundPlane& source) const noexcept {
    Eigen::Matrix3f basis_source;
    basis_source.col(0) = source.t1;
    basis_source.col(1) = source.t2;
    basis_source.col(2) = source.n;

    Eigen::Matrix3f basis_target;
    basis_target.col(0) = t1;
    basis_target.col(1) = t2;
    basis_target.col(2) = n;

    const Eigen::Matrix3f R = basis_target.transpose() * basis_source;
    const Eigen::Vector3f t = basis_target.transpose() * (source.centroid - centroid);
    const Eigen::Matrix2f A = R.topLeftCorner<2, 2>();
    const Eigen::Vector2f b = t.head<2>();
    const Eigen::Vector2f s = R.topRightCorner<2, 1>();
    const Eigen::RowVector2f dx = R.bottomLeftCorner<1, 2>();

    const float ds = R(2, 2);
    const float db = t.z();

    return [A, b, s, dx, ds, db](Eigen::Vector2f& point, float& distance) {
        const Eigen::Vector2f point_prev = point;
        const float distance_prev = distance;
        point = A * point_prev + b + distance_prev * s;
        distance = dx.dot(point_prev) + ds * distance_prev + db;
    };
}

Eigen::Vector3f GroundPlane::project(const Eigen::Vector3f& point) const noexcept {
    Eigen::Vector3f local = point - centroid;
    return {t1.dot(local), t2.dot(local), local.dot(n)};
}