#include "odometry/projection.h"
#include "odometry/ground_plane.h"

// system
#include <tuple>
#include <utility>

namespace {
void correspond(const VoxelMap<2, BirdsEyeCloud>& source, const VoxelMap<2, BirdsEyeCloud>& target, std::vector<std::tuple<const Eigen::Vector2f&, const Eigen::Vector2f&, float>>& correspondences) {
    correspondences.clear();
    tbb::enumerable_thread_specific<std::vector<std::tuple<const Eigen::Vector2f&, const Eigen::Vector2f&, float>>> correspondence_pool;
    tbb::parallel_for(source.view(), [&target, &correspondence_pool](auto& range) {
        for(const auto& [query, data] : range) {
            auto neighbor = target.closest(query, data);
            if(!neighbor) continue;
            auto& local = correspondence_pool.local();
            local.emplace_back(query, neighbor->point, neighbor->distance_sq);
        }
    });
    for(auto& local : correspondence_pool) {
        for(auto& [query, point, distance_sq] : local) {
            correspondences.emplace_back(query, point, distance_sq);
        }
    }
}
} // private scope

void Projection::fit(const VoxelMap<3, SemanticCloud>& map, const Sophus::SE3f& pose, const Sophus::SE3f& T) {
    // update projection
    auto T_icp_projected = ground_.project(T);
    target_.apply(std::move(T_icp_projected));
    // recompute ground
    if((origin_ - pose.translation()).squaredNorm() > (max_distance_ground_ * max_distance_ground_ * 0.25f)) {
        origin_ = pose.translation();
        auto T_g2g = ground_.update(map, origin_, pred_ground_);
        // reproject
        target_.apply(T_g2g);
    }
}

std::optional<Sophus::SE3f> Projection::align(const SemanticCloud& source, size_t max_iterations, float convergence_criterion) {
    BirdsEyeCloud points{source, ground_, pred_target_};
    if(target_.empty()) {
        target_.extend(points);
        return std::nullopt;
    }
    // construct source map (current frame)
    source_.clear();
    source_.extend(points);
    // chi termination criterion
    float chi = -1.f;
    // preallocate correspondence vector
    std::vector<std::tuple<const Eigen::Vector2f&, const Eigen::Vector2f&, float>> correspondences;
    // build transform
    Sophus::SE2f T{};
    size_t iter;
    for(iter = 0; iter < max_iterations; ++iter) {
        // build bidirectional correspondences
        correspond(source_, target_, correspondences);
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
    // determine ratio of inliers (bidirectional)
    // correspond(source_, target_, correspondences);
    // const size_t source_inlier_count = std::count_if(correspondences.begin(), correspondences.end(),
    //     [inlier_distance_sq](const auto& correspondence) {
    //         const auto& [source_point, target_point, distance_sq] = correspondence;
    //         return distance_sq <= inlier_distance_sq;
    //     });
    // correspond(target_, source_, correspondences);
    // const size_t target_inlier_count = std::count_if(correspondences.begin(), correspondences.end(),
    //     [inlier_distance_sq](const auto& correspondence) {
    //         const auto& [target_point, source_point, distance_sq] = correspondence;
    //         return distance_sq <= inlier_distance_sq;
    //     });
    // reject the solution if it required max_iterations OR there was a bad fit
    if(iter == max_iterations || correspondences.size() * 2 < std::ranges::distance(source_.view())) { // || target_inlier_count * 2 < std::ranges::distance(target_.view())) {
        target_.clear();
        target_.extend(points);
        return std::nullopt;
    }
    target_ = std::move(source_);
    return std::make_optional(ground_.lift(T));
}
