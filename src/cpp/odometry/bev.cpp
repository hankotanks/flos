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
        search.value().emplace_back(projection_.size());
        projection_.emplace_back(local_projection);
    }
    BEV::compute_centroids(voxel_size_, voxel_capacity / 2); // TODO: 2 is a magic number
}

void BEV::correspondences(const BEV& target, std::vector<size_t>& indices, float correspondence_distance_sq) const {
    indices.resize(centroids.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, centroids.size()), [this, &target, &indices, correspondence_distance_sq](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            const Eigen::Vector2f& centroid = this->centroids[i];
            float best_dist_sq = correspondence_distance_sq;
            size_t best_index = std::numeric_limits<size_t>::max();
            for(size_t j = 0; j < target.centroids.size(); ++j) {
                const float dist_sq = (centroid - target.centroids[j]).squaredNorm();
                if(dist_sq >= best_dist_sq) continue;
                best_dist_sq = dist_sq;
                best_index = j;
            }
            indices[i] = best_index;
        }
    });
}

inline float voxel_lower_bound_distance_sq(const Eigen::Vector2f& query, const Eigen::Vector2i& voxel, float voxel_size) {
    const float h = 0.5f * voxel_size;
    const Eigen::Vector2f center = voxel_to_point(voxel, voxel_size);
    Eigen::Vector2f d = (query - center).cwiseAbs() - Eigen::Vector2f::Constant(h);
    d = d.cwiseMax(Eigen::Vector2f::Zero());
    return d.squaredNorm();
}

void BEV::correspondences_full(const BEV& target, std::vector<size_t>& indices, float correspondence_distance_sq) const {
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
    indices.resize(projection_.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, projection_.size()), [this, &target, &indices, correspondence_distance_sq](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            float best_dist_sq = correspondence_distance_sq;
            size_t best_index = std::numeric_limits<size_t>::max();
            const Eigen::Vector2f& point = this->projection_[i];
            Eigen::Vector2i key = point_to_voxel(point, voxel_size_);
            for(const auto& offset : offsets) {
                const Eigen::Vector2i key_offset = key + offset;
                float lb_distance_sq = voxel_lower_bound_distance_sq(point, key_offset, voxel_size_);
                if(lb_distance_sq >= best_dist_sq) continue;
                auto search = target.index_.find(key_offset);
                if(search == target.index_.end()) continue;
                const std::vector<size_t>& indices = search.value();
                for(const size_t& j : indices) {
                    float distance_sq = (target.projection_[j] - point).squaredNorm();
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
    if(centroids.empty() || target.centroids.empty()) return std::nullopt;
    float chi = -1.f;
    float correspondence_distance_sq = correspondence_distance * correspondence_distance;
    // preallocate vector correspondence vector
    // std::vector<size_t> source_correspondences(centroids.size());
    std::vector<size_t> target_correspondences(target.projection_.size());
    // build transform
    Sophus::SE2f T{};
    for(size_t iter = 0; iter < max_iterations; ++iter) {
        // build bidirectional correspondences
        // BEV::correspondences(target, source_correspondences, correspondence_distance_sq);
        target.correspondences_full(*this, target_correspondences, correspondence_distance_sq);
        // ensure symmetry and construct means
        Eigen::Vector2f source_mean = Eigen::Vector2f::Zero();
        Eigen::Vector2f target_mean = Eigen::Vector2f::Zero();
        float chi_i = 0.f;
        size_t count = 0;
        for(size_t i = 0; i < target.projection_.size(); ++i) {
            const size_t j = target_correspondences[i];
            if(j == std::numeric_limits<size_t>::max()) continue;
            const Eigen::Vector2f& source_point = projection_[j];
            const Eigen::Vector2f& target_point = target.projection_[i];
            source_mean += source_point;
            target_mean += target_point;
            chi_i += (projection_[j] - target.projection_[i]).squaredNorm();
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
        for(size_t i = 0; i < target.projection_.size(); ++i) {
            const size_t j = target_correspondences[i];
            if(j == std::numeric_limits<size_t>::max()) continue;
            const Eigen::Vector2f& source_point = projection_[j];
            const Eigen::Vector2f& target_point = target.projection_[i];
            const Eigen::Vector2f source_norm = source_point - source_mean;
            const Eigen::Vector2f target_norm = target_point - target_mean;
            dot += source_norm.dot(target_norm);
            cross += source_norm.x() * target_norm.y() - source_norm.y() * target_norm.x();
        }
        // perturbation for this iteration
        const Sophus::SO2f R{std::atan2(cross, dot)};
        const Sophus::SE2f est{R, target_mean - R * source_mean};
        // shift centroids
        for(Eigen::Vector2f& c : projection_) c = est * c;
        // accumulate transform
        T = est * T;
        // termination criterion
        if(chi > 0.f && std::abs(chi - chi_i) <= convergence_criterion * chi) break;
        chi = chi_i;
    }
    return T;
}

void BEV::transform(const Sophus::SE2f& T) {
    std::for_each(projection_.begin(), projection_.end(), [&T](Eigen::Vector2f& p) { p = T * p; });
    std::for_each(centroids.begin(), centroids.end(), [&T](Eigen::Vector2f& p) { p = T * p; });
}

void BEV::compute_centroids(float eps, size_t min_pts) {
    if(projection_.empty()) return;
    const std::size_t n = projection_.size();

    // for comparison against squaredNorm
    const float eps_sq = eps * eps;

    // cell neighborhood
    static const std::array<Eigen::Vector2i, 9> offsets{
        Eigen::Vector2i{ 0,  0},
        Eigen::Vector2i{ 1,  0},
        Eigen::Vector2i{-1,  0},
        Eigen::Vector2i{ 0,  1},
        Eigen::Vector2i{ 0, -1},
        Eigen::Vector2i{ 1,  1},
        Eigen::Vector2i{ 1, -1},
        Eigen::Vector2i{-1,  1},
        Eigen::Vector2i{-1, -1},
    };
    // build adjacency graph
    std::vector<std::vector<std::size_t>> neighbors(n);
    tbb::parallel_for_each(index_.cbegin(), index_.cend(), [this, &neighbors, eps_sq, min_pts](const auto& entry) {
        const auto& [key, indices] = entry;
        for(size_t i : indices) {
            neighbors[i].reserve(min_pts);
            for(const auto& offset : offsets) {
                auto search = this->index_.find(key + offset);
                if(search == this->index_.end()) continue;
                const std::vector<size_t>& query_indices = search.value();
                for(size_t j : query_indices) {
                    if((this->projection_[i] - this->projection_[j]).squaredNorm() > eps_sq) continue;
                    neighbors[i].emplace_back(j);
                }
            }
        }
    });

    // designate core points
    std::vector<bool> core(n, false);
    tbb::parallel_for<size_t>(0, n, [&](std::size_t i) { core[i] = neighbors[i].size() >= min_pts; });

    // construct clusters
    static constexpr int LABEL_UNVISITED = -1;
    static constexpr int LABEL_NOISE = -2;
    std::vector<int> labels(n, LABEL_UNVISITED);
    int cluster_id = 0;
    for(size_t i = 0; i < n; ++i) {
        if(labels[i] != LABEL_UNVISITED) continue;
        // mark non-core points as noise temporarily
        if(!core[i]) {
            labels[i] = LABEL_NOISE;
            continue;
        }
        // set label
        labels[i] = cluster_id;
        // process the cluster
        std::vector<std::size_t> queue{neighbors[i]};
        size_t j = 0;
        while(j < queue.size()) {
            size_t k = queue[j++];
            // noise points are eligible to become non-core points in the current cluster
            if(labels[k] == LABEL_NOISE) labels[k] = cluster_id;
            if(labels[k] != LABEL_UNVISITED) continue;
            labels[k] = cluster_id;
            if(!core[k]) continue;
            // extend the cluster is k is a core point
            for(size_t k : neighbors[k]) {
                if(labels[k] != LABEL_UNVISITED && labels[k] != LABEL_NOISE) continue;
                queue.emplace_back(k);
            }
        }
        ++cluster_id;
    }

    // compute centroids
    size_t cluster_count = static_cast<size_t>(cluster_id);
    std::vector<Eigen::Vector2f> cluster_accumulator(cluster_count, Eigen::Vector2f::Zero());
    std::vector<std::size_t> cluster_sizes(cluster_count, 0);
    // accumulate clusters
    for(size_t i = 0; i < n; ++i) {
        const int label = labels[i];
        if(label < 0) continue;
        cluster_accumulator[static_cast<size_t>(label)] += projection_[i];
        ++cluster_sizes[static_cast<size_t>(label)];
    }
    // average to find centroids
    centroids.reserve(cluster_count);
    for(int c = 0; c < cluster_id; ++c) {
        size_t id = static_cast<size_t>(c);
        if(cluster_sizes[id] == 0) continue;
        centroids.emplace_back(cluster_accumulator[id] / static_cast<float>(cluster_sizes[id]));
    }
}