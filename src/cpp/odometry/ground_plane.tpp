#pragma once

#include "odometry/ground_plane.h"

#include <tbb/parallel_for.h>

template<typename F>
template<typename Predicate>
GroundPlane<F>::GroundPlane(const VoxelMap<3, F>& map, const Eigen::Vector3f& origin, float max_distance, Predicate&& pred) {
    tbb::enumerable_thread_specific<std::pair<Eigen::Vector3f, size_t>> centroid_pool;
    tbb::parallel_for(map.view_within_radius(origin, max_distance), [&centroid_pool](auto& range) {
        for(const auto& [point, data] : range) {
            if(data.prediction != 1) continue;
            auto& local = centroid_pool.local();
            local.first += point;
            local.second++;
        }
    });
    size_t count = 0;
    for(const auto& [local_centroid, local_count] : centroid_pool)  {
        centroid += local_centroid;
        count += local_count;
    }
    centroid /= static_cast<float>(count);
    // covariance
    tbb::enumerable_thread_specific<Eigen::Matrix3f> covariance_pool;
    tbb::parallel_for(map.view_within_radius(origin, max_distance), [this, &covariance_pool](auto& range) {
        for(const auto& [point, data] : range) {
            if(data.prediction != 1) continue;
            auto& local = covariance_pool.local();
            const Eigen::Vector3f d = point - this->centroid;
            local += d * d.transpose();
        }
    });
    Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
    for(const auto& local_covariance : covariance_pool) S += local_covariance;
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
Sophus::SE3f GroundPlane<F>::lift(const Sophus::SE2f& T) const noexcept {
    Eigen::Matrix3f R_plane = Eigen::Matrix3f::Identity();
    R_plane.block<2, 2>(1, 1) = T.so2().matrix();
    const Eigen::Matrix3f R = plane_ * R_plane * plane_.transpose();
    const Eigen::Vector3f t = centroid + plane_.block<3, 2>(0, 1) * T.translation() - R * centroid;
    return Sophus::SE3f{R, t};
}

template<typename F>
Sophus::SE2f GroundPlane<F>::project(const Sophus::SE3f& T) const noexcept {
    const Eigen::Matrix3f R = T.so3().matrix();
    const Eigen::Vector3f t = T.translation();
    const Eigen::Matrix<float, 3, 2> B = plane_.block<3, 2>(0, 1);
    const Eigen::Matrix2f A = B.transpose() * R * B;
    Eigen::JacobiSVD<Eigen::Matrix2f> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2f R2 = svd.matrixU() * svd.matrixV().transpose();
    if(R2.determinant() < 0.f) {
        auto U = svd.matrixU();
        U.col(1) *= -1.f;
        R2 = U * svd.matrixV().transpose();
    }
    Eigen::Matrix3f R_plane = Eigen::Matrix3f::Identity();
    R_plane.block<2, 2>(1, 1) = R2;
    const Eigen::Matrix3f R_projected = plane_ * R_plane * plane_.transpose();
    const Eigen::Vector2f t2 = B.transpose() * (t + R_projected * centroid - centroid);
    return Sophus::SE2f{R2, t2};
}

template<typename F>
Eigen::Vector2f GroundPlane<F>::project(const Eigen::Vector3f& point) const noexcept {
    Eigen::Vector3f local = point - centroid;
    return {t1.dot(local), t2.dot(local)};
}