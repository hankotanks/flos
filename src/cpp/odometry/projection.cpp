#include "odometry/projection.h"
#include <tuple>
#include <utility>

namespace {
void correspond(const VoxelMap<2>& source, const VoxelMap<2>& target, std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector2f, float>>& correspondences, float correspondence_distance_sq) {
    correspondences.clear();
    tbb::enumerable_thread_specific<std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector2f, float>>> correspondence_pool;
    tbb::parallel_for(source.view(), [&target, &correspondence_distance_sq, &correspondence_pool](auto& range) {
        for(const auto& query : range) {
            auto neighbor = target.closest(query, correspondence_distance_sq);
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

std::optional<Sophus::SE2f> Projection::align(size_t max_iterations, float convergence_criterion, float correspondence_distance) {
    if(curr.empty() || prev.empty()) return std::nullopt;
    float chi = -1.f;
    float correspondence_distance_sq = correspondence_distance * correspondence_distance;
    // preallocate correspondence vector
    std::vector<std::tuple<Eigen::Vector2f, Eigen::Vector2f, float>> correspondences;
    // build transform
    Sophus::SE2f T{};
    for(size_t iter = 0; iter < max_iterations; ++iter) {
        // build bidirectional correspondences
        correspond(curr, prev, correspondences, correspondence_distance_sq);
        if(correspondences.size() < 3) return std::nullopt;
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
        curr.apply(est);
        // accumulate transform
        T = est * T;
        // termination criterion
        if(chi > 0.f && std::abs(chi - chi_i) <= convergence_criterion * chi) break;
        chi = chi_i;
    }
    return T;
}

template<typename Predicate>
BirdsEyeCloud::BirdsEyeCloud(const SemanticCloud& points, const GroundPlane<SemanticCloud>& ground, Predicate&& pred) {
    for(size_t i = 0; i < points.size(); ++i) {
        auto [point, data] = points[i];
        if(!pred(data)) continue;
        points_.emplace_back(ground.project(point));
    }
}

std::pair<Eigen::Vector2f, float> BirdsEyeCloud::operator[](size_t i) const noexcept {
    const Eigen::Vector3f& point = points_[i];
    return std::pair<Eigen::Vector2f, float>(std::piecewise_construct, std::forward_as_tuple(point.x(), point.y()), std::forward_as_tuple(point.z()));
}
