#include "odometry/bev.h"

#include <Eigen/Eigenvalues>
#include <limits>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_for_each.h>
#include <tbb/parallel_for_each.h>
#include <utility>

namespace {
inline Eigen::Vector2i point_to_voxel(const Eigen::Vector2f& point, double voxel_size) {
    return {
        static_cast<int>(std::floor(point.x() / voxel_size)),
        static_cast<int>(std::floor(point.y() / voxel_size))
    };
}

inline Eigen::Vector2f voxel_to_point(const Eigen::Vector2i& voxel, double voxel_size) {
    return {
        (static_cast<float>(voxel.x()) + 0.5f) * static_cast<float>(voxel_size),
        (static_cast<float>(voxel.y()) + 0.5f) * static_cast<float>(voxel_size)
    };
}
} // private scope

BEV::BEV(const Eigen::Vector3f& centroid, const Eigen::Matrix3f& ground, const std::span<const Eigen::Vector3f>& points, const std::span<const uint8_t>& semantics, float voxel_size, size_t voxel_capacity) : 
    voxel_size_(voxel_size) {
    Eigen::Vector3f t1 = ground.col(1);
    Eigen::Vector3f t2 = ground.col(2);
    for(size_t i = 0; i < points.size(); ++i) {
        if(semantics[i] != 3) continue;
        Eigen::Vector3f local = points[i] - centroid;
        Eigen::Vector2f local_projection{t1.dot(local), t2.dot(local)};
        const Eigen::Vector2i voxel = point_to_voxel(local_projection, voxel_size_);
        auto search = index_.find(voxel);
        if(search == index_.end()) {
            auto [it, _] = index_.emplace(std::piecewise_construct,
                std::forward_as_tuple(voxel),
                std::forward_as_tuple());
            search = it;
        }
        search.value().emplace(points_projected.size());
        points_projected.emplace_back(local_projection);
    }
}

inline float voxel_lower_bound_distance_sq(const Eigen::Vector2f& query, const Eigen::Vector2i& voxel, float voxel_size) {
    const float h = 0.5f * voxel_size;
    const Eigen::Vector2f center = voxel_to_point(voxel, voxel_size);
    Eigen::Vector2f d = (query - center).cwiseAbs() - Eigen::Vector2f::Constant(h);
    d = d.cwiseMax(Eigen::Vector2f::Zero());
    return d.squaredNorm();
}

void BEV::correspondences(const BEV& target, std::vector<size_t>& indices, float correspondence_distance_sq) const {
    static const std::array<Eigen::Vector2i, 9> offsets{
        Eigen::Vector2i{ 0,  0},
        Eigen::Vector2i{ 1,  0},
        Eigen::Vector2i{-1,  0},
        Eigen::Vector2i{ 0,  1},
        Eigen::Vector2i{ 0, -1},
        Eigen::Vector2i{-1, -1},
        Eigen::Vector2i{-1,  1},
        Eigen::Vector2i{ 1,  1},
        Eigen::Vector2i{ 1, -1},
    };
    indices.resize(points_projected.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, points_projected.size()), [this, &target, &indices, correspondence_distance_sq](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            float best_dist_sq = correspondence_distance_sq;
            size_t best_index = std::numeric_limits<size_t>::max();
            const Eigen::Vector2f& point = this->points_projected[i];
            Eigen::Vector2i key = point_to_voxel(point, voxel_size_);
            for(const auto& offset : offsets) {
                const Eigen::Vector2i key_offset = key + offset;
                float lb_distance_sq = voxel_lower_bound_distance_sq(point, key_offset, voxel_size_);
                if(lb_distance_sq >= best_dist_sq) continue;
                auto search = target.index_.find(key_offset);
                if(search == target.index_.end()) continue;
                const std::set<size_t>& indices = search.value();
                for(const size_t& j : indices) {
                    float distance_sq = (target.points_projected[j] - point).squaredNorm();
                    if(distance_sq >= correspondence_distance_sq) continue;
                    if(distance_sq >= best_dist_sq) continue;
                    best_dist_sq = distance_sq;
                    best_index = j;
                }
            }
            indices[i] = best_index;
        }
    });
}

std::optional<Sophus::SE2f> BEV::align(const BEV& target, size_t max_iterations, float convergence_criterion, float correspondence_distance) {
    if(points_projected.empty() || target.points_projected.empty()) return std::nullopt;
    float chi = -1.f;
    float correspondence_distance_sq = correspondence_distance * correspondence_distance;
    // preallocate correspondence vector
    std::vector<size_t> target_correspondences(target.points_projected.size());
    // build transform
    Sophus::SE2f T{};
    for(size_t iter = 0; iter < max_iterations; ++iter) {
        // build bidirectional correspondences
        // BEV::correspondences(target, source_correspondences, correspondence_distance_sq);
        target.correspondences(*this, target_correspondences, correspondence_distance_sq);
        // ensure symmetry and construct means
        Eigen::Vector2f source_mean = Eigen::Vector2f::Zero();
        Eigen::Vector2f target_mean = Eigen::Vector2f::Zero();
        float chi_i = 0.f;
        size_t count = 0;
        for(size_t i = 0; i < target.points_projected.size(); ++i) {
            const size_t j = target_correspondences[i];
            if(j == std::numeric_limits<size_t>::max()) continue;
            source_mean += points_projected[j];
            target_mean += target.points_projected[i];
            chi_i += (points_projected[j] - target.points_projected[i]).squaredNorm();
            count++;
        }
        if(count < 3) return std::nullopt;
        // if(count < 3) return std::nullopt; // break early if there are not enough correspondences
        const float count_inv = 1.f / static_cast<float>(count);
        source_mean *= count_inv;
        target_mean *= count_inv;
        chi_i *= count_inv;
        // compute optimal rotation directly
        float dot = 0.f;
        float cross = 0.f;
        for(size_t i = 0; i < target.points_projected.size(); ++i) {
            const size_t j = target_correspondences[i];
            if(j == std::numeric_limits<size_t>::max()) continue;
            const Eigen::Vector2f source_norm = points_projected[j] - source_mean;
            const Eigen::Vector2f target_norm = target.points_projected[i] - target_mean;
            dot += source_norm.dot(target_norm);
            cross += source_norm.x() * target_norm.y() - source_norm.y() * target_norm.x();
        }
        // perturbation for this iteration
        const Sophus::SO2f R{std::atan2(cross, dot)};
        const Sophus::SE2f est{R, target_mean - R * source_mean};
        // shift centroids
        BEV::transform(est);
        // accumulate transform
        T = est * T;
        // termination criterion
        if(chi > 0.f && std::abs(chi - chi_i) <= convergence_criterion * chi) break;
        chi = chi_i;
    }
    return T;
}

void BEV::transform(const Sophus::SE2f& T) {
    for(size_t i = 0; i < points_projected.size(); ++i) {
        Eigen::Vector2f& point = points_projected[i];
        Eigen::Vector2i key_before = point_to_voxel(point, voxel_size_);
        point = T * point;
        Eigen::Vector2i key_after = point_to_voxel(point, voxel_size_);
        if(key_before != key_after) {
            index_[key_before].erase(i);
            auto search = index_.find(key_after);
            if(search == index_.end()) {
                auto [it, _] = index_.emplace(std::piecewise_construct,
                    std::forward_as_tuple(key_after),
                    std::forward_as_tuple());
                search = it;
            }
            search.value().emplace(i);
        }
    }
}
