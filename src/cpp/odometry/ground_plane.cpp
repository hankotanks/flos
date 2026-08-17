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

// TODO: this is insufficient because we don't currently store the distance above the plane
// we can do this, but its probably simpler to just reproject the points
// see the note below...
std::function<Eigen::Vector2f(const Eigen::Vector2f&, float)> GroundPlane::project(const Sophus::SE3f& T) const noexcept {
    const Eigen::Matrix3f& R = T.so3().matrix();
    const Eigen::Vector3f& t = T.translation();
    Eigen::Matrix<float, 3, 2> basis;
    basis.col(0) = t1;
    basis.col(1) = t2;
    Eigen::Affine2f T2 = Eigen::Affine2f::Identity();
    T2.linear() = basis.transpose() * R * basis;
    T2.translation() = basis.transpose() * (R * centroid + t - centroid);
    Eigen::Vector2f t_prime = basis.transpose() * R * plane_.col(0);
    // ... this affine transform is correct, however, we need to consider another translational component
    // Eigen::Vector2f t_prime = basis.transpose() * R * plane_.col(0);
    // in order to actually compute a new points position, we need to do the following:
    // T2 * x_i + s_i * t_prime
    // where x_i is the 2D point and s_i is the signed distance from the plane.
    // this can be stored as point_data in the VoxelMap, but it requires another wrapper implementation
    // this could be defined in projection.h, but I really don't think it's worth it
    return [T2, t_prime](const Eigen::Vector2f& point, const float& s) -> Eigen::Vector2f { return T2 * point + s * t_prime; };
}

std::function<Eigen::Vector2f(const Eigen::Vector2f&, float)> GroundPlane::project(const GroundPlane& source) const noexcept {
    Eigen::Matrix3f source_basis;
    source_basis.col(0) = source.t1;
    source_basis.col(1) = source.t2;
    source_basis.col(2) = source.n;

    Eigen::Matrix3f target_basis;
    target_basis.col(0) = t1;
    target_basis.col(1) = t2;
    target_basis.col(2) = n;

    const Eigen::Matrix3f R = target_basis.transpose() * source_basis;

    const Eigen::Vector3f translation =
        target_basis.transpose() * (source.centroid - centroid);

    const Eigen::Matrix<float, 2, 2> A =
        R.topLeftCorner<2, 2>();

    const Eigen::Vector2f b =
        translation.head<2>();

    const Eigen::Vector2f s =
        R.topRightCorner<2, 1>();

    return [A, b, s](
        const Eigen::Vector2f& point,
        float distance
    ) -> Eigen::Vector2f {
        return A * point + b + distance * s;
    };
}

Eigen::Vector3f GroundPlane::project(const Eigen::Vector3f& point) const noexcept {
    Eigen::Vector3f local = point - centroid;
    return {t1.dot(local), t2.dot(local), local.dot(n)};
}