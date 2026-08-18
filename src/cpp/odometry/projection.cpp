#include "odometry/projection.h"
#include <tuple>
#include <utility>

namespace {
void correspond(const VoxelMap<2, BirdsEyeCloud>& source, const VoxelMap<2, BirdsEyeCloud>& target, std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector2f, float>>& correspondences, float correspondence_distance_sq) {
    correspondences.clear();
    tbb::enumerable_thread_specific<std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector2f, float>>> correspondence_pool;
    tbb::parallel_for(source.view(), [&target, &correspondence_distance_sq, &correspondence_pool](auto& range) {
        for(const auto& [query, data] : range) {
            auto neighbor = target.closest(query, data, correspondence_distance_sq);
            if(!neighbor) continue;
            auto& local = correspondence_pool.local();
            local.emplace_back(query, neighbor->point, neighbor->distance_sq);
        }
    });
    for(auto& local : correspondence_pool) {
        correspondences.insert(correspondences.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
    }
}
} // private scope

std::optional<Sophus::SE2f> Projection::align(const BirdsEyeCloud& points, size_t max_iterations, float convergence_criterion, float correspondence_distance) {
    if(target_.empty()) {
        target_.extend(points);
        return std::nullopt;
    }
    // construct source map (current frame)
    source_.clear();
    source_.extend(points);
    // chi termination criterion
    float chi = -1.f;
    float correspondence_distance_sq = correspondence_distance * correspondence_distance;
    // preallocate correspondence vector
    std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector2f, float>> correspondences;
    // build transform
    Sophus::SE2f T{};
    for(size_t iter = 0; iter < max_iterations; ++iter) {
        // build bidirectional correspondences
        correspond(source_, target_, correspondences, correspondence_distance_sq);
        if(correspondences.size() < 3) {
            target_.clear();
            target_.extend(points);
            return std::nullopt;
        }
        // ensure symmetry and construct means
        Eigen::Vector2f source_mean = Eigen::Vector2f::Zero();
        Eigen::Vector2f target_mean = Eigen::Vector2f::Zero();
        float chi_i = 0.f;
        for(const auto& [source_point, target_point, distance_sq] : correspondences) {
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
        for(const auto& [source_point, target_point, distance_sq] : correspondences) {
            const Eigen::Vector2f source_norm = source_point - source_mean;
            const Eigen::Vector2f target_norm = target_point - target_mean;
            dot += source_norm.dot(target_norm);
            cross += source_norm.x() * target_norm.y() - source_norm.y() * target_norm.x();
        }
        // perturbation for this iteration
        const Sophus::SO2f R{std::atan2(cross, dot)};
        const Sophus::SE2f est{R, target_mean - R * source_mean};
        // shift centroids
        source_.apply(est);
        // accumulate transform
        T = est * T;
        // termination criterion
        if(chi > 0.f && std::abs(chi - chi_i) <= convergence_criterion * chi) break;
        chi = chi_i;
    }
    target_ = std::move(source_);
    return T;
}
