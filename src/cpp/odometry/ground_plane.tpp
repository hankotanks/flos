#pragma once

#include "odometry/ground_plane.h"

// system
#include <limits>
// tbb
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

// TODO: is this possible without allocating the points again?
// not a big cost, since its only done once every ~5 meters
template<typename Predicate>
std::function<void(Eigen::Vector2f&, float&)> GroundPlane::update(const VoxelMap<3, SemanticCloud>& map, const Eigen::Vector3f& origin, Predicate&& pred) {
    // point candidates
    tbb::enumerable_thread_specific<std::vector<Eigen::Vector3f>> point_pool([] { return std::vector<Eigen::Vector3f>{}; });
    tbb::parallel_for(map.points_within_radius(origin, max_distance_), [&pred, &point_pool](auto& range) {
        auto& local = point_pool.local();
        for(const auto& [point, data] : range) {
            if(!pred(data)) continue;
            local.push_back(point);
        }
    });
    // reserve a vector for the ground points
    std::vector<Eigen::Vector3f> points;
    size_t count = 0;
    for(const auto& local : point_pool) count += local.size();
    points.reserve(count);
    // populate it
    for(const auto& local : point_pool) points.insert(points.end(), local.begin(), local.end());
    if(points.size() < 3) return [](Eigen::Vector2f&, float&) { /* STUB */ };

    // represents a consesnsus
    struct Consensus {
        Eigen::Vector3f point{Eigen::Vector3f::Zero()};
        Eigen::Vector3f normal{Eigen::Vector3f::Zero()};
        float error{0.f};
        size_t count{0};
    };

    Consensus best;
    best.error = std::numeric_limits<float>::max();

    std::mt19937 rng{static_cast<std::mt19937::result_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count())};

    // get random index generator
    std::uniform_int_distribution<size_t> distribution(0, points.size() - 1);
    for(size_t i = 0, j0 = distribution(rng); i < iterations_; ++i, j0 = distribution(rng)) {
        size_t j1, j2;
        do { j1 = distribution(rng); } while (j1 == j0);
        do { j2 = distribution(rng); } while (j2 == j0 || j2 == j1);
        const Eigen::Vector3f& p0 = points[j0];
        const Eigen::Vector3f& p1 = points[j1];
        const Eigen::Vector3f& p2 = points[j2];
        // construct and test current candidate
        Consensus current;
        current.point = p0;
        current.normal = (p1 - p0).cross(p2 - p0);
        // manually normalize the... normal
        auto s = current.normal.squaredNorm();
        if(s < 1e-6) continue;
        current.normal /= std::sqrt(s);
        // get inlier count and error
        for(const Eigen::Vector3f& point : points) {
            const float distance = current.normal.dot(point - p0);
            if(std::abs(distance) <= inlier_distance_) {
                current.count++;
                current.error += distance * distance;
            }
        }
        if(current.count < best.count) continue;
        if(current.count == best.count && current.error > best.error) continue;
        best = current;
    }
    if(best.count < 3) return [](Eigen::Vector2f&, float&) { /* STUB */ };

    // compute centroid and covariance
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    {
        size_t count = 0;
        for(const Eigen::Vector3f& point : points) {
            if(std::abs(best.normal.dot(point - best.point)) > inlier_distance_) continue;
            centroid += point;
            ++count;
        }
        if(count < 3) return [](Eigen::Vector2f&, float&) { /* STUB */ };
        centroid /= static_cast<float>(count);
        // S
        for(const Eigen::Vector3f& point : points) {
            const float distance = best.normal.dot(point - best.point);
            if(std::abs(distance) > inlier_distance_) continue;
            const Eigen::Vector3f d = point - centroid;
            covariance += d * d.transpose();
        }
        covariance /= static_cast<float>(count);
    }
    
    // get plane
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver{covariance};
    Eigen::Matrix<float, 3, 3> plane = solver.eigenvectors();
    Eigen::Vector3f n  = plane.col(0);
    Eigen::Vector3f t1 = plane.col(1);
    Eigen::Vector3f t2 = plane.col(2);
    // normal orientation
    if(n.dot(n_) < 0.f) {
        n = -n;
        t2 = -t2;
    }
    // preserve the orientation of the projection x-axis
    if(t1.dot(t1_) < 0.f) {
        t1 = -t1;
        t2 = -t2;
    }
    // ensure correct handedness
    if(t1.cross(t2).dot(n) < 0.f) t2 = -t2;
    // rebuild the basis after all sign corrections
    plane.col(0) = n;
    plane.col(1) = t1;
    plane.col(2) = t2;

    // get transform to update projection
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

    // update plane
    centroid_ = centroid;
    n_ = n;
    t1_ = t1;
    t2_ = t2;
    plane_ = plane;

    // return transformation
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