#include "odometry/bev.h"

// eigen
#include <Eigen/Eigenvalues>
// tbb
#include <tbb/parallel_for_each.h>
#include <tbb/enumerable_thread_specific.h>

BEV::BEV(const Eigen::Vector3f& centroid, const Eigen::Matrix3f& ground, const std::span<const Eigen::Vector3f>& points, const std::span<const uint8_t>& semantics, float max_distance, float voxel_size, size_t voxel_capacity) : projection(max_distance, voxel_size, voxel_capacity) {
    Eigen::Vector3f t1 = ground.col(1);
    Eigen::Vector3f t2 = ground.col(2);
    for(size_t i = 0; i < points.size(); ++i) {
        if(semantics[i] != 3) continue;
        Eigen::Vector3f local = points[i] - centroid;
        projection.extend({t1.dot(local), t2.dot(local)}, 0);
    }
}

void BEV::correspond(const BEV& target, std::vector<BEV::Correspondence>& correspondences, float correspondence_distance_sq) const {
    correspondences.clear();
    tbb::enumerable_thread_specific<std::vector<Correspondence>> local;
    projection.for_each([&](const Eigen::Vector2f& query, const uint8_t& data) {
        auto neighbor = target.projection.closest(query, data, correspondence_distance_sq);
        if(neighbor) local.local().emplace_back(query, neighbor->point, neighbor->distance_sq);
    });
    for(auto& v : local) {
        correspondences.insert(correspondences.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
    }
}

std::optional<Sophus::SE2f> BEV::align(const BEV& target, size_t max_iterations, float convergence_criterion, float correspondence_distance) {
    if(projection.empty() || target.projection.empty()) return std::nullopt;
    float chi = -1.f;
    float correspondence_distance_sq = correspondence_distance * correspondence_distance;
    // preallocate correspondence vector
    std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector2f, float>> correspondences;
    // build transform
    Sophus::SE2f T{};
    for(size_t iter = 0; iter < max_iterations; ++iter) {
        // build bidirectional correspondences
        target.correspond(*this, correspondences, correspondence_distance_sq);
        if(correspondences.size() < 3) return std::nullopt;
        // ensure symmetry and construct means
        Eigen::Vector2f source_mean = Eigen::Vector2f::Zero();
        Eigen::Vector2f target_mean = Eigen::Vector2f::Zero();
        float chi_i = 0.f;
        for(const auto& [target_point, source_point, distance_sq] : correspondences) {
            source_mean += source_point;
            target_mean += target_point;
            chi_i += distance_sq;
        }
        const float count_inv = 1.f / static_cast<float>(correspondences.size());
        source_mean *= count_inv;
        target_mean *= count_inv;
        chi_i *= count_inv;
        // compute optimal rotation directly
        float dot = 0.f;
        float cross = 0.f;
        for(const auto& [target_point, source_point, distance_sq] : correspondences) {
            const Eigen::Vector2f source_norm = source_point - source_mean;
            const Eigen::Vector2f target_norm = target_point - target_mean;
            dot += source_norm.dot(target_norm);
            cross += source_norm.x() * target_norm.y() - source_norm.y() * target_norm.x();
        }
        // perturbation for this iteration
        const Sophus::SO2f R{std::atan2(cross, dot)};
        const Sophus::SE2f est{R, target_mean - R * source_mean};
        // shift centroids
        projection.transform(est);
        // accumulate transform
        T = est * T;
        // termination criterion
        if(chi > 0.f && std::abs(chi - chi_i) <= convergence_criterion * chi) break;
        chi = chi_i;
    }
    return T;
}

void BEV::transform(const Sophus::SE2f& T) {
    projection.transform(T);
}
