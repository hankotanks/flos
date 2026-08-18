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
    tbb::enumerable_thread_specific<std::pair<Eigen::Vector3f, size_t>> centroid_pool([] {
        return std::make_pair(Eigen::Vector3f::Zero(), size_t{0});
    });
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
    tbb::enumerable_thread_specific<Eigen::Matrix3f> covariance_pool([] { 
        return Eigen::Matrix3f::Zero();
    });
    tbb::parallel_for(map.view_within_radius(origin, max_distance), [this, &pred, &covariance_pool](auto& range) {
        for(const auto& [point, data] : range) {
            if(!pred(data)) continue;
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
    n = plane_.col(0);
    t1 = plane_.col(1);
    t2 = plane_.col(2);
    if(t1.cross(t2).dot(n) < 0.f) {
        plane_.col(2) *= -1.f;
        t2 *= -1.f;
    }
}

template<typename Predicate>
GroundPlane::GroundPlane(
    const VoxelMap<3, SemanticCloud>& map,
    const Eigen::Vector3f& origin,
    float max_distance,
    Predicate&& pred,
    float inlier_distance,
    size_t ransac_iterations)
{
    // -------------------------------------------------------------------------
    // 1. Gather candidate points
    // -------------------------------------------------------------------------

    tbb::enumerable_thread_specific<std::vector<Eigen::Vector3f>> point_pool([] {
        return std::vector<Eigen::Vector3f>{};
    });

    tbb::parallel_for(
        map.view_within_radius(origin, max_distance),
        [&pred, &point_pool](auto& range) {
            auto& local = point_pool.local();

            for (const auto& [point, data] : range) {
                if (!pred(data))
                    continue;

                local.push_back(point);
            }
        });

    std::vector<Eigen::Vector3f> points;

    size_t total_count = 0;
    for (const auto& local : point_pool)
        total_count += local.size();

    points.reserve(total_count);

    for (const auto& local : point_pool)
        points.insert(points.end(), local.begin(), local.end());

    if (points.size() < 3) {
        throw std::runtime_error("Not enough points for RANSAC plane fitting");
    }

    // -------------------------------------------------------------------------
    // 2. RANSAC
    // -------------------------------------------------------------------------

    std::mt19937 rng{
        static_cast<std::mt19937::result_type>(
            std::chrono::high_resolution_clock::now()
                .time_since_epoch()
                .count())
    };

    std::uniform_int_distribution<size_t> distribution(
        0, points.size() - 1);

    Eigen::Vector3f best_normal = Eigen::Vector3f::Zero();
    Eigen::Vector3f best_point = Eigen::Vector3f::Zero();

    size_t best_inlier_count = 0;
    float best_error = std::numeric_limits<float>::max();

    const float threshold_sq = inlier_distance * inlier_distance;

    for (size_t iteration = 0; iteration < ransac_iterations; ++iteration) {
        // Pick three distinct points.
        const size_t i0 = distribution(rng);

        size_t i1;
        do {
            i1 = distribution(rng);
        } while (i1 == i0);

        size_t i2;
        do {
            i2 = distribution(rng);
        } while (i2 == i0 || i2 == i1);

        const Eigen::Vector3f& p0 = points[i0];
        const Eigen::Vector3f& p1 = points[i1];
        const Eigen::Vector3f& p2 = points[i2];

        const Eigen::Vector3f v1 = p1 - p0;
        const Eigen::Vector3f v2 = p2 - p0;

        Eigen::Vector3f normal = v1.cross(v2);

        const float normal_sq = normal.squaredNorm();

        // Degenerate / nearly collinear sample.
        if (normal_sq < 1e-8f)
            continue;

        normal /= std::sqrt(normal_sq);

        // Plane equation:
        //
        //     normal.dot(x - p0) = 0
        //
        // Squared point-to-plane distance:
        //
        //     (normal.dot(x - p0))^2
        //

        size_t inlier_count = 0;
        float error = 0.f;

        for (const Eigen::Vector3f& point : points) {
            const float distance =
                normal.dot(point - p0);

            const float distance_sq = distance * distance;

            if (distance_sq <= threshold_sq) {
                ++inlier_count;
                error += distance_sq;
            }
        }

        // Prefer more inliers, then prefer the tighter fit.
        if (inlier_count > best_inlier_count ||
            (inlier_count == best_inlier_count && error < best_error)) {

            best_inlier_count = inlier_count;
            best_error = error;
            best_normal = normal;
            best_point = p0;
        }
    }

    if (best_inlier_count < 3) {
        throw std::runtime_error("RANSAC failed to find a plane");
    }

    // -------------------------------------------------------------------------
    // 3. Refine using ONLY RANSAC inliers
    // -------------------------------------------------------------------------

    centroid = Eigen::Vector3f::Zero();

    size_t count = 0;

    for (const Eigen::Vector3f& point : points) {
        const float distance =
            best_normal.dot(point - best_point);

        if (std::abs(distance) <= inlier_distance) {
            centroid += point;
            ++count;
        }
    }

    if (count < 3) {
        throw std::runtime_error("Not enough RANSAC inliers");
    }

    centroid /= static_cast<float>(count);

    // -------------------------------------------------------------------------
    // 4. PCA refinement on the RANSAC inliers
    // -------------------------------------------------------------------------

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();

    for (const Eigen::Vector3f& point : points) {
        const float distance =
            best_normal.dot(point - best_point);

        if (std::abs(distance) > inlier_distance)
            continue;

        const Eigen::Vector3f d = point - centroid;
        covariance += d * d.transpose();
    }

    covariance /= static_cast<float>(count);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver{
        covariance
    };

    plane_ = solver.eigenvectors();

    n  = plane_.col(0);
    t1 = plane_.col(1);
    t2 = plane_.col(2);

    if (t1.cross(t2).dot(n) < 0.f) {
        plane_.col(2) *= -1.f;
        t2 *= -1.f;
    }
}
