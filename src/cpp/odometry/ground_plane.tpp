#pragma once

#include "odometry/ground_plane.h"

template<typename F>
template<typename Predicate>
GroundPlane<F>::GroundPlane(const VoxelMap<3, F>& map, const Eigen::Vector3f& origin, float max_distance, Predicate&& pred) {
    // centroid
    struct Acc {
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        size_t count{0};
        Acc& operator+=(const Acc& rhs) {
            centroid += rhs.centroid;
            count += rhs.count;
            return *this;
        }
    };
    auto [centroid_temp, count] = map.reduce_within_radius(origin, max_distance, Acc{}, 
        [&pred](Acc acc, const Eigen::Vector3f& point, const typename F::point_data_t& data) -> Acc {
            if(!pred(data)) return acc;
            const auto [centroid, count] = acc;
            return {centroid + point, count + 1};
        });
    centroid = centroid_temp / static_cast<float>(count);
    // covariance
    Eigen::Matrix3f S = map.reduce_within_radius(origin, max_distance, Eigen::Matrix3f::Zero().eval(), 
        [this, &pred](Eigen::Matrix3f acc, const Eigen::Vector3f& point, const typename F::point_data_t& data) -> Eigen::Matrix3f {
            if(!pred(data)) return acc;
            const Eigen::Vector3f d = point - this->centroid;
            return acc + d * d.transpose();
        });
    S /= static_cast<float>(count);
    // solve
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver{S};
    plane_ = solver.eigenvectors();
    t1 = plane_.col(1);
    t2 = plane_.col(2);
    if(t1.cross(t2).dot(plane_.col(0)) < 0.f) {
        plane_.col(2) *= -1.f;
        t2 *= -1.f;
    }
}

template<typename F>
Eigen::Vector2f GroundPlane<F>::project(const Eigen::Vector3f& point) const noexcept {
    Eigen::Vector3f local = point - centroid;
    return {t1.dot(local), t2.dot(local)};
}

template<typename F>
Sophus::SE3f GroundPlane<F>::lift(const Sophus::SE2f& T) const noexcept {
    Eigen::Matrix3f R_plane = Eigen::Matrix3f::Identity();
    R_plane.block<2, 2>(1, 1) = T.so2().matrix();
    const Eigen::Matrix3f R = plane_ * R_plane * plane_.transpose();
    const Eigen::Vector3f t = centroid + plane_.block<3, 2>(0, 1) * T.translation() - R * centroid;
    return Sophus::SE3f{R, t};
}