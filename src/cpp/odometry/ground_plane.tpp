#pragma once

#include "odometry/ground_plane.h"

#include <tbb/parallel_for.h>

template<typename Predicate>
BirdsEyeCloud::BirdsEyeCloud(const SemanticCloud& points, const GroundPlane& ground, Predicate&& pred) {
    for(size_t i = 0; i < points.size(); ++i) {
        auto [point, data] = points[i];
        if(!pred(data)) continue;
        points_.emplace_back(ground.project(point));
    }
}

template<typename Predicate>
GroundPlane::GroundPlane(const VoxelMap<3, SemanticCloud>& map, const Eigen::Vector3f& origin, float max_distance, Predicate&& pred) {
    tbb::enumerable_thread_specific<std::pair<Eigen::Vector3f, size_t>> centroid_pool;
    tbb::parallel_for(map.view_within_radius(origin, max_distance), [&pred, &centroid_pool](auto& range) {
        for(const auto& [point, data] : range) {
            if(!pred(data)) continue;
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
