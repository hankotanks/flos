#include "odometry/bev.h"

#include <Eigen/Eigenvalues>
#include <oneapi/tbb/parallel_for.h>


namespace dbscan2d {
struct Cell {
    int x, y;
    bool operator==(const Cell& other) const { return x == other.x && y == other.y; }
};

struct CellHash {
    std::size_t operator()(const Cell& c) const noexcept {
        const std::uint64_t x = static_cast<std::uint32_t>(c.x);
        const std::uint64_t y = static_cast<std::uint32_t>(c.y);
        std::uint64_t h = x * 0x9e3779b97f4a7c15ULL;
        h ^= y + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return static_cast<std::size_t>(h);
    }
};

inline Cell cell_of(const Eigen::Vector2f& p, float cell_size) noexcept {
    return {
        static_cast<int>(std::floor(p.x() / cell_size)),
        static_cast<int>(std::floor(p.y() / cell_size))
    };
}

std::vector<Eigen::Vector2f> cluster_centroids(const std::vector<Eigen::Vector2f>& points, float eps, int min_pts) {
    if(points.empty()) return {};

    const std::size_t n = points.size();
    const float eps2 = eps * eps;

    std::unordered_map<Cell, std::vector<std::size_t>, CellHash> grid;
    grid.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        grid[cell_of(points[i], eps)].push_back(i);
    }

    // ------------------------------------------------------------------
    // 2. Compute all eps-neighborhoods in parallel.
    //
    // Each point writes only to its own neighbors[i], so this is
    // race-free.
    // ------------------------------------------------------------------

    std::vector<std::vector<std::size_t>> neighbors(n);

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n),
        [&](const tbb::blocked_range<std::size_t>& range) {
            for (std::size_t i = range.begin(); i != range.end(); ++i) {
                const Cell c = cell_of(points[i], eps);

                // Reserve a little space to avoid repeated allocations.
                neighbors[i].reserve(static_cast<std::size_t>(min_pts));

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const Cell nc{c.x + dx, c.y + dy};

                        auto it = grid.find(nc);
                        if (it == grid.end()) {
                            continue;
                        }

                        for (const std::size_t j : it->second) {
                            const Eigen::Vector2f d = points[i] - points[j];

                            if (d.squaredNorm() <= eps2) {
                                neighbors[i].push_back(j);
                            }
                        }
                    }
                }
            }
        });

    // ------------------------------------------------------------------
    // 3. Identify core points.
    // ------------------------------------------------------------------

    std::vector<bool> core(n, false);

    for (std::size_t i = 0; i < n; ++i) {
        core[i] = neighbors[i].size() >= static_cast<std::size_t>(min_pts);
    }

    // ------------------------------------------------------------------
    // 4. Standard DBSCAN cluster expansion.
    //
    // This part is intentionally sequential. DBSCAN's expansion has
    // dependencies between points, and parallelizing it usually adds
    // synchronization without much benefit.
    // ------------------------------------------------------------------

    constexpr int UNVISITED = -1;
    constexpr int NOISE = -2;

    std::vector<int> labels(n, UNVISITED);

    int cluster_id = 0;

    for (std::size_t i = 0; i < n; ++i) {
        if (labels[i] != UNVISITED) {
            continue;
        }

        // Not a core point -> temporarily mark as noise.
        if (!core[i]) {
            labels[i] = NOISE;
            continue;
        }

        labels[i] = cluster_id;

        std::vector<std::size_t> queue;
        queue.reserve(neighbors[i].size());

        for (const std::size_t j : neighbors[i]) {
            queue.push_back(j);
        }

        std::size_t head = 0;

        while (head < queue.size()) {
            const std::size_t j = queue[head++];

            // A point previously classified as noise can become a
            // border point of this cluster.
            if (labels[j] == NOISE) {
                labels[j] = cluster_id;
            }

            if (labels[j] != UNVISITED) {
                continue;
            }

            labels[j] = cluster_id;

            // Only core points expand the cluster.
            if (!core[j]) {
                continue;
            }

            for (const std::size_t k : neighbors[j]) {
                if (labels[k] == UNVISITED || labels[k] == NOISE) {
                    queue.push_back(k);
                }
            }
        }

        ++cluster_id;
    }

    // ------------------------------------------------------------------
    // 5. Compute one centroid per cluster.
    //
    // Border points are included, noise is excluded.
    // ------------------------------------------------------------------

    std::vector<Eigen::Vector2f> sums(
        static_cast<std::size_t>(cluster_id),
        Eigen::Vector2f::Zero());

    std::vector<std::size_t> counts(
        static_cast<std::size_t>(cluster_id),
        0);

    for (std::size_t i = 0; i < n; ++i) {
        const int label = labels[i];

        if (label >= 0) {
            sums[static_cast<std::size_t>(label)] += points[i];
            ++counts[static_cast<std::size_t>(label)];
        }
    }

    std::vector<Eigen::Vector2f> centroids;
    centroids.reserve(static_cast<std::size_t>(cluster_id));

    for (int c = 0; c < cluster_id; ++c) {
        if (counts[static_cast<std::size_t>(c)] == 0) {
            continue;
        }

        centroids.push_back(
            sums[static_cast<std::size_t>(c)] /
            static_cast<float>(counts[static_cast<std::size_t>(c)]));
    }

    return centroids;
}
} // namespace dbscan2d

BEV::BEV(const Eigen::Vector3f& centroid, const Eigen::Matrix3f& ground, const std::span<const Eigen::Vector3f>& points, const std::span<const uint8_t>& semantics, float max_distance, float voxel_size, size_t voxel_capacity) {
    float max_distance_sq = max_distance * max_distance;
    Eigen::Vector3f t1 = ground.col(1);
    Eigen::Vector3f t2 = ground.col(2);
    for(size_t i = 0; i < points.size(); ++i) {
        if(semantics[i] != 3) continue;
        Eigen::Vector3f local = points[i] - centroid;
        Eigen::Vector2f local_proj{t1.dot(local), t2.dot(local)};
        if(local_proj.squaredNorm() > max_distance_sq) continue;
        projection.emplace_back(local_proj);
    }
    centroids = dbscan2d::cluster_centroids(projection, voxel_size, voxel_capacity / 2);
}

#if 0
Sophus::SE2f BEV::align(const BEV& target, float cell_size) {
    if(centroids.empty() || target.centroids.empty()) return Sophus::SE2f();

    const float max_d2 = cell_size * cell_size * 4.f;

    struct Cell {
        int x, y;
        bool operator==(const Cell& o) const {
            return x == o.x && y == o.y;
        }
    };

    struct Hash {
        std::size_t operator()(const Cell& c) const {
            return std::hash<int>{}(c.x) ^
                   (std::hash<int>{}(c.y) << 1);
        }
    };

    auto cell = [cell_size](const Eigen::Vector2f& p) {
        return Cell{
            static_cast<int>(std::floor(p.x() / cell_size)),
            static_cast<int>(std::floor(p.y() / cell_size))
        };
    };

    std::unordered_map<Cell, std::vector<std::size_t>, Hash> grid;
    for (std::size_t i = 0; i < target.centroids.size(); ++i)
        grid[cell(target.centroids[i])].push_back(i);

    Sophus::SE2f T;

    for (int iter = 0; iter < 500; ++iter) {
        std::vector<Eigen::Vector2f> src_corr;
        std::vector<Eigen::Vector2f> tgt_corr;

        src_corr.reserve(centroids.size());
        tgt_corr.reserve(centroids.size());

        // ------------------------------------------------------------
        // Find all candidate correspondences.
        // ------------------------------------------------------------

        struct Match {
            std::size_t src;
            std::size_t tgt;
            float d2;

            bool operator<(const Match& other) const {
                return d2 < other.d2;
            }
        };

        std::vector<Match> matches;
        matches.reserve(centroids.size() * 4);

        for (std::size_t i = 0; i < centroids.size(); ++i) {
            const Eigen::Vector2f p = T * centroids[i];
            const Cell c = cell(p);

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const Cell nc{c.x + dx, c.y + dy};

                    const auto it = grid.find(nc);
                    if (it == grid.end())
                        continue;

                    for (const std::size_t j : it->second) {
                        const float d2 =
                            (p - target.centroids[j]).squaredNorm();

                        if (d2 < max_d2) {
                            matches.push_back({
                                i,
                                j,
                                d2
                            });
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------
        // Greedy one-to-one assignment.
        // Closest pairs get priority.
        // ------------------------------------------------------------

        std::sort(matches.begin(), matches.end());

        std::vector<bool> src_used(centroids.size(), false);
        std::vector<bool> tgt_used(target.centroids.size(), false);

        for (const Match& m : matches) {
            if (src_used[m.src])
                continue;

            if (tgt_used[m.tgt])
                continue;

            src_used[m.src] = true;
            tgt_used[m.tgt] = true;

            src_corr.push_back(T * centroids[m.src]);
            tgt_corr.push_back(target.centroids[m.tgt]);
        }

        if (src_corr.size() < 2)
            break;

        // ------------------------------------------------------------
        // Estimate dT: tgt_corr ~= dT * src_corr
        // ------------------------------------------------------------

        Eigen::Vector2f src_mean = Eigen::Vector2f::Zero();
        Eigen::Vector2f tgt_mean = Eigen::Vector2f::Zero();

        for (std::size_t i = 0; i < src_corr.size(); ++i) {
            src_mean += src_corr[i];
            tgt_mean += tgt_corr[i];
        }

        src_mean /= static_cast<float>(src_corr.size());
        tgt_mean /= static_cast<float>(tgt_corr.size());

        Eigen::Matrix2f H = Eigen::Matrix2f::Zero();

        for (std::size_t i = 0; i < src_corr.size(); ++i) {
            H += (src_corr[i] - src_mean) *
                (tgt_corr[i] - tgt_mean).transpose();
        }

        Eigen::JacobiSVD<Eigen::Matrix2f> svd(
            H,
            Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::Matrix2f U = svd.matrixU();
        Eigen::Matrix2f V = svd.matrixV();

        Eigen::Matrix2f R = V * U.transpose();

        if (R.determinant() < 0.0f) {
            V.col(1) *= -1.0f;
            R = V * U.transpose();
        }

        const Eigen::Vector2f t =
            tgt_mean - R * src_mean;

        const float theta =
            std::atan2(R(1, 0), R(0, 0));

        const Sophus::SE2f dT(
            Sophus::SO2f(theta),
            t);

        T = dT * T;

        if (dT.translation().norm() < 1e-4f &&
            std::abs(dT.so2().log()) < 1e-4f)
            break;
    }

    return T;
}
#else
Sophus::SE2f BEV::align(const BEV& target, float cell_size) {
    if (centroids.empty() || target.centroids.empty())
        return Sophus::SE2f();

    const float max_d2 = cell_size * cell_size * 4.f;

    struct Cell {
        int x, y;

        bool operator==(const Cell& o) const {
            return x == o.x && y == o.y;
        }
    };

    struct Hash {
        std::size_t operator()(const Cell& c) const {
            return std::hash<int>{}(c.x) ^
                   (std::hash<int>{}(c.y) << 1);
        }
    };

    auto cell = [cell_size](const Eigen::Vector2f& p) {
        return Cell{
            static_cast<int>(std::floor(p.x() / cell_size)),
            static_cast<int>(std::floor(p.y() / cell_size))
        };
    };

    // ------------------------------------------------------------
    // Target spatial grid.
    // ------------------------------------------------------------

    std::unordered_map<Cell, std::vector<std::size_t>, Hash> target_grid;

    for (std::size_t j = 0; j < target.centroids.size(); ++j)
        target_grid[cell(target.centroids[j])].push_back(j);

    Sophus::SE2f T;

    for (int iter = 0; iter < 500; ++iter) {
        std::vector<Eigen::Vector2f> src_corr;
        std::vector<Eigen::Vector2f> tgt_corr;

        src_corr.reserve(centroids.size());
        tgt_corr.reserve(centroids.size());

        // ------------------------------------------------------------
        // Transform source points using current estimate.
        // ------------------------------------------------------------

        std::vector<Eigen::Vector2f> transformed_src(
            centroids.size());

        for (std::size_t i = 0; i < centroids.size(); ++i)
            transformed_src[i] = T * centroids[i];

        // ------------------------------------------------------------
        // Find best target for every source.
        //
        // source_best[i] = target index, or npos if none is close enough.
        // ------------------------------------------------------------

        const std::size_t npos = std::numeric_limits<std::size_t>::max();

        std::vector<std::size_t> source_best(
            centroids.size(), npos);

        std::vector<float> source_best_d2(
            centroids.size(), std::numeric_limits<float>::max());

        for (std::size_t i = 0; i < transformed_src.size(); ++i) {
            const Eigen::Vector2f& p = transformed_src[i];
            const Cell c = cell(p);

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const Cell nc{c.x + dx, c.y + dy};

                    const auto it = target_grid.find(nc);
                    if (it == target_grid.end())
                        continue;

                    for (const std::size_t j : it->second) {
                        const float d2 =
                            (p - target.centroids[j]).squaredNorm();

                        if (d2 < max_d2 &&
                            d2 < source_best_d2[i]) {
                            source_best_d2[i] = d2;
                            source_best[i] = j;
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------
        // Build a spatial grid of the TRANSFORMED source points.
        //
        // This allows us to perform the reverse query:
        // for every target point, find its nearest source point.
        // ------------------------------------------------------------

        std::unordered_map<Cell, std::vector<std::size_t>, Hash>
            source_grid;

        for (std::size_t i = 0; i < transformed_src.size(); ++i)
            source_grid[cell(transformed_src[i])].push_back(i);

        // ------------------------------------------------------------
        // Find best source for every target.
        //
        // target_best[j] = source index, or npos if none is close enough.
        // ------------------------------------------------------------

        std::vector<std::size_t> target_best(
            target.centroids.size(), npos);

        std::vector<float> target_best_d2(
            target.centroids.size(),
            std::numeric_limits<float>::max());

        for (std::size_t j = 0; j < target.centroids.size(); ++j) {
            const Eigen::Vector2f& q = target.centroids[j];
            const Cell c = cell(q);

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const Cell nc{c.x + dx, c.y + dy};

                    const auto it = source_grid.find(nc);
                    if (it == source_grid.end())
                        continue;

                    for (const std::size_t i : it->second) {
                        const float d2 =
                            (transformed_src[i] - q).squaredNorm();

                        if (d2 < max_d2 &&
                            d2 < target_best_d2[j]) {
                            target_best_d2[j] = d2;
                            target_best[j] = i;
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------
        // Mutual nearest-neighbor correspondences.
        //
        // Accept (i,j) only if:
        //
        //     best target of i == j
        //     best source of j == i
        // ------------------------------------------------------------

        for (std::size_t i = 0; i < centroids.size(); ++i) {
            const std::size_t j = source_best[i];

            if (j == npos)
                continue;

            if (target_best[j] != i)
                continue;

            src_corr.push_back(transformed_src[i]);
            tgt_corr.push_back(target.centroids[j]);
        }

        // ------------------------------------------------------------
        // Need at least two correspondences to estimate SE(2).
        //
        // With fewer than two, do not attempt the SVD.
        // Return the estimate accumulated so far.
        // ------------------------------------------------------------

        if (src_corr.size() < 2)
            break;

        // ------------------------------------------------------------
        // Estimate dT:
        //
        //     tgt_corr ~= dT * src_corr
        // ------------------------------------------------------------

        Eigen::Vector2f src_mean = Eigen::Vector2f::Zero();
        Eigen::Vector2f tgt_mean = Eigen::Vector2f::Zero();

        for (std::size_t i = 0; i < src_corr.size(); ++i) {
            src_mean += src_corr[i];
            tgt_mean += tgt_corr[i];
        }

        src_mean /= static_cast<float>(src_corr.size());
        tgt_mean /= static_cast<float>(tgt_corr.size());

        Eigen::Matrix2f H = Eigen::Matrix2f::Zero();

        for (std::size_t i = 0; i < src_corr.size(); ++i) {
            H +=
                (src_corr[i] - src_mean) *
                (tgt_corr[i] - tgt_mean).transpose();
        }

        Eigen::JacobiSVD<Eigen::Matrix2f> svd(
            H,
            Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::Matrix2f U = svd.matrixU();
        Eigen::Matrix2f V = svd.matrixV();

        Eigen::Matrix2f R = V * U.transpose();

        if (R.determinant() < 0.0f) {
            V.col(1) *= -1.0f;
            R = V * U.transpose();
        }

        const Eigen::Vector2f t =
            tgt_mean - R * src_mean;

        const float theta =
            std::atan2(R(1, 0), R(0, 0));

        const Sophus::SE2f dT(
            Sophus::SO2f(theta),
            t);

        T = dT * T;

        // ------------------------------------------------------------
        // Convergence.
        // ------------------------------------------------------------

        if (dT.translation().norm() < 1e-4f &&
            std::abs(dT.so2().log()) < 1e-4f) {
            break;
        }
    }

    return T;
}
#endif

void BEV::transform(const Sophus::SE2f& T) {
    std::for_each(projection.begin(), projection.end(), [&T](Eigen::Vector2f& p) { p = T * p; });
    std::for_each(centroids.begin(), centroids.end(), [&T](Eigen::Vector2f& p) { p = T * p; });
}