#pragma once

#include "odometry/ground_plane.h"

#include <tbb/parallel_for.h>

template<typename Predicate>
std::function<void(Eigen::Vector2f&, float&)> GroundPlane::update(const VoxelMap<3, SemanticCloud>& map, const Eigen::Vector3f& origin, Predicate&& pred) {
    // point candidates
    tbb::enumerable_thread_specific<std::vector<Eigen::Vector3f>> point_pool([] { return std::vector<Eigen::Vector3f>{}; });
    tbb::parallel_for(map.view_within_radius(origin, max_distance_), [&pred, &point_pool](auto& range) {
        auto& local = point_pool.local();
        for(const auto& [point, data] : range) {
            if(!pred(data)) continue;
            local.push_back(point);
        }
    });
    std::vector<Eigen::Vector3f> points;
    size_t count_total = 0;
    for(const auto& local : point_pool) count_total += local.size();
    points.reserve(count_total);
    for(const auto& local : point_pool) points.insert(points.end(), local.begin(), local.end());
    if(points.size() < 3) return [](Eigen::Vector2f&, float&) { /* STUB */ };

    // ransac
    std::mt19937 rng{static_cast<std::mt19937::result_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count())};
    std::uniform_int_distribution<size_t> distribution(0, points.size() - 1);

    Eigen::Vector3f best_normal = Eigen::Vector3f::Zero();
    Eigen::Vector3f best_point = Eigen::Vector3f::Zero();
    float best_error = std::numeric_limits<float>::max();
    size_t best_count_inlier = 0;

    const float threshold_sq = inlier_distance_ * inlier_distance_;
    for(size_t i = 0, j0 = distribution(rng); i < iterations_; ++i, j0 = distribution(rng)) {
        size_t j1, j2;
        do { j1 = distribution(rng); } while (j1 == j0);
        do { j2 = distribution(rng); } while (j2 == j0 || j2 == j1);

        const Eigen::Vector3f& p0 = points[j0];
        const Eigen::Vector3f& p1 = points[j1];
        const Eigen::Vector3f& p2 = points[j2];

        const Eigen::Vector3f v1 = p1 - p0;
        const Eigen::Vector3f v2 = p2 - p0;

        Eigen::Vector3f normal = v1.cross(v2);
        const float normal_sq = normal.squaredNorm();
        if(normal_sq < 1e-8f) continue;
        normal /= std::sqrt(normal_sq);

        size_t count_inlier = 0;
        float error = 0.f;
        for(const Eigen::Vector3f& point : points) {
            const float distance = normal.dot(point - p0);
            const float distance_sq = distance * distance;
            if(distance_sq <= threshold_sq) {
                ++count_inlier;
                error += distance_sq;
            }
        }

        if(count_inlier > best_count_inlier || (count_inlier == best_count_inlier && error < best_error)) {
            best_count_inlier = count_inlier;
            best_error = error;
            best_normal = normal;
            best_point = p0;
        }
    }
    if(best_count_inlier < 3) return [](Eigen::Vector2f&, float&) { /* STUB */ };

    // compute centroid
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    size_t count = 0;
    for(const Eigen::Vector3f& point : points) {
        const float distance = best_normal.dot(point - best_point);
        if(std::abs(distance) <= inlier_distance_) {
            centroid += point;
            ++count;
        }
    }
    if(count < 3) return [](Eigen::Vector2f&, float&) { /* STUB */ };
    centroid /= static_cast<float>(count);

    // PCA on inliers
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for(const Eigen::Vector3f& point : points) {
        const float distance = best_normal.dot(point - best_point);
        if(std::abs(distance) > inlier_distance_) continue;
        const Eigen::Vector3f d = point - centroid;
        covariance += d * d.transpose();
    }
    covariance /= static_cast<float>(count);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver{covariance};
    Eigen::Matrix<float, 3, 3> plane = solver.eigenvectors();
    Eigen::Vector3f n  = plane.col(0);
    Eigen::Vector3f t1 = plane.col(1);
    Eigen::Vector3f t2 = plane.col(2);
    if(t1.cross(t2).dot(n) < 0.f) {
        plane.col(2) *= -1.f;
        t2 *= -1.f;
    }

    Eigen::Matrix3f basis_source;
    basis_source.col(0) = t1_;
    basis_source.col(1) = t2_;
    basis_source.col(2) = n_;

    Eigen::Matrix3f basis_target;
    basis_target.col(0) = t1;
    basis_target.col(1) = t2;
    basis_target.col(2) = n;

    const Eigen::Matrix3f R = basis_target.transpose() * basis_source;
    const Eigen::Vector3f t = basis_target.transpose() * (centroid_ - centroid);
    const Eigen::Matrix2f A = R.topLeftCorner<2, 2>();
    const Eigen::Vector2f b = t.head<2>();
    const Eigen::Vector2f s = R.topRightCorner<2, 1>();
    const Eigen::RowVector2f dx = R.bottomLeftCorner<1, 2>();

    const float ds = R(2, 2);
    const float db = t.z();

    centroid_ = centroid;
    n_ = n;
    t1_ = t1;
    t2_ = t2;
    plane_ = plane;

    return [A, b, s, dx, ds, db](Eigen::Vector2f& point, float& distance) {
        const Eigen::Vector2f point_prev = point;
        const float distance_prev = distance;
        point = A * point_prev + b + distance_prev * s;
        distance = dx.dot(point_prev) + ds * distance_prev + db;
    };
}

template<typename Predicate>
BirdsEyeCloud::BirdsEyeCloud(const SemanticCloud& points, const GroundPlane& ground, Predicate&& pred) {
    for(size_t i = 0; i < points.size(); ++i) {
        auto [point, data] = points[i];
        if(!pred(data)) continue;
        points_.emplace_back(ground.project(point));
    }
}
