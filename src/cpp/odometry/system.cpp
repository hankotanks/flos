#include "odometry/system.h"

// system
#include <algorithm>
#include <limits>
#include <optional>
#include <span>
// tbb
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
// package
#include "odometry/bev.h"
#include "classes.h"
// util
#include "util/cloud.h"
#include "util/profiler.h"

namespace {
// field names to search in get_stamp_field
static constexpr std::array<std::string, 4> STAMP_FIELD_NAMES = {"timestamps", "time", "timestamp", "t"};

template<typename T>
std::optional<std::string> get_stamp_field(util::Cloud& points) noexcept {
    for(const std::string& name : STAMP_FIELD_NAMES) {
        try {
            (void) points.field<T>(name);
        } catch(...) {
            continue;
        }
        return name;
    }
    return std::nullopt;
}

template<typename T>
void deskew_frame_inner(util::Cloud& points, const std::string& name, const Sophus::SE3f& motion) noexcept {
    std::span<const T> stamps = points.field<T>(name);
    const auto& [t_min, t_max] = std::minmax_element(stamps.begin(), stamps.end());
    const auto t_norm = [&](const T t) { 
        return (t - *t_min) / (*t_max - *t_min); 
    };
    const auto& omega = motion.log();
    tbb::parallel_for(tbb::blocked_range<size_t>{0, points.size()}, [&](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            const T stamp = t_norm(stamps[i]);
            const auto pose = Sophus::SE3f::exp((static_cast<float>(stamp) - 1.f) * omega);
            points[i] = pose * points[i];
        };
    });
}

void deskew_frame(util::Cloud& points, const Sophus::SE3f& motion) noexcept {
    auto name = get_stamp_field<float>(points);
    if(name) {
        deskew_frame_inner<float>(points, *name, motion);
        return;
    }
    name = get_stamp_field<double>(points);
    if(name) deskew_frame_inner<double>(points, *name, motion);
}
} // private scope

namespace {
inline float linear_velocity(const Eigen::Matrix<float,6,1>& xi, float dt) {
    return xi.tail<3>().norm() / dt;
}

inline Eigen::Vector3f angular_wrap(const Eigen::Vector3f& rot) {
    double angle = rot.norm();
    if(angle < 1e-12) return rot;
    double wrapped = std::fmod(angle + M_PI, 2.0 * M_PI) - M_PI;
    return rot * (wrapped / angle);
}

inline float angular_velocity(const Eigen::Matrix<float,6,1>& xi, float dt) {
    Eigen::Vector3f omega = angular_wrap(xi.head<3>()) / dt;
    return omega.norm();
}
} // private scope

OdometrySystem::OdometrySystem(const Config& config) : util::Profiler("OdometrySystem"),
    config_(config),
    map_(config.max_distance, config.voxel_size, config.voxel_capacity),
    threshold_(config.max_distance, config.threshold.initial, config.threshold.motion),
    experimental(config.max_distance, config.voxel_size, config.voxel_capacity) {
    size_t max_threads = config.max_threads > 0 ? \
        config.max_threads : tbb::this_task_arena::max_concurrency();
    volatile static const tbb::global_control _ = tbb::global_control( // NOTE: static storage type is required here
        tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_threads)); 
}

const Sophus::SE3f& OdometrySystem::register_to_map(util::Cloud& points) noexcept {
    config_.fixed_dt = true;
    return OdometrySystem::register_to_map(points, 0, 0);
}

const Sophus::SE3f& OdometrySystem::register_to_map(util::Cloud& points, unsigned int seconds, unsigned int nanoseconds) noexcept {
    // reset statistics
    statistics = Statistics{};
    // perform registration
    PROFILE(registration, this) {
        // compute delta
        const auto& [delta, dt] = OdometrySystem::predict_delta(seconds, nanoseconds);
        // deskewing
        if(config_.deskew) {
            PROFILE(deskew, &registration) {
                deskew_frame(points, delta);
            }
        }
        // downsample incoming points to obtain keypoints and the preprocessed frame
        util::Cloud points_kp{points};
        PROFILE(downsample, &registration) {
            points_kp.downsample(config_.voxel_size * OdometrySystem::VOXEL_SIZE_SCALAR);
        }
        // compute pose w/ ICP
        SemanticCloud points_kp_wrapper{points_kp};
        const auto [pose, deviation] = OdometrySystem::align(points_kp_wrapper, delta, registration);
        // normalize statistics
        PROFILE_SKIP(registration) {
            // get mean correspondences
            statistics.correspondences /= static_cast<float>(statistics.iterations);
            statistics.threshold.min /= static_cast<float>(statistics.iterations);
            statistics.threshold.max /= static_cast<float>(statistics.iterations);
        }
        // update AdaptiveThreshold
        threshold_.update(deviation);
        // append points to map
        PROFILE(update, &registration) {
            SemanticCloud points_wrapper{points};
            map_.update(points_wrapper, pose);
        }
        // store new pose
        // update last pose
        last_delta_ = last_pose_.inverse() * pose;
        last_pose_ = pose;
        // if we aren't handling variable dt, theres no need to update stamps or velocity statistics
        PROFILE_SKIP(registration) {
            // log final velocity statistics
            Eigen::Matrix<float, 6, 1> xi = last_delta_.log();
            statistics.velocity.linear = linear_velocity(xi, dt);
            statistics.velocity.angular = angular_velocity(xi, dt);
        }
    }
    // return const ref to the computed pose, since we store it anyway
    return last_pose_;
}

std::pair<Sophus::SE3f, double> OdometrySystem::predict_delta(unsigned int seconds, unsigned int nanoseconds) noexcept {
    // handling depends on whether we rely on timestamps
    if(config_.fixed_dt) {
        return std::make_pair(last_delta_, 1.f);
    } else {
        if(init_seconds_ == std::numeric_limits<unsigned int>::max()) init_seconds_ = seconds;
        double stamp = static_cast<double>(seconds - init_seconds_) + 1e-9 * static_cast<double>(nanoseconds);
        double dt_prev = last_stamp_ - prev_stamp_;
        double dt_curr = stamp - last_stamp_;
        double scale = 1.0;
        if(dt_prev > 1e-6) scale = static_cast<double>(dt_curr / dt_prev);
        // update timestamp state
        prev_stamp_ = last_stamp_;
        last_stamp_ = stamp;
        // compute predicted delta:
        return std::make_pair(Sophus::SE3d::exp(scale * last_delta_.cast<double>().log()).cast<float>(), dt_curr);
    }
}

OdometrySystem::CorrespondenceSet OdometrySystem::correspond(const SemanticCloud& points, float correspondence_distance) const noexcept {
    tbb::concurrent_vector<Correspondence> correspondences;
    correspondences.reserve(points.size());
    // collect all correspondences
    float correspondence_distance_sq = correspondence_distance * correspondence_distance;
    tbb::parallel_for(tbb::blocked_range<size_t>{0, points.size()}, [this, &points, &correspondences, correspondence_distance_sq](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            const auto& [point, point_data] = points[i];
            const auto neighbor = this->map_.closest(point, point_data, correspondence_distance_sq);
            if(neighbor) correspondences.emplace_back(point, point_data, *neighbor);
        }
    });
    // prune all low-weight correspondences down to a threshold...
    if(config_.correspondence.target > 0 && config_.correspondence.target < correspondences.size()) {
        std::nth_element(correspondences.begin(), correspondences.begin() + config_.correspondence.target, correspondences.end(),
            [](const Correspondence& fst, const Correspondence& snd) {
                const auto& [fst_point, fst_data, fst_neighbor] = fst;
                const auto& [snd_point, snd_data, snd_neighbor] = snd;
                return fst_neighbor.weight > snd_neighbor.weight;
            });
        correspondences.resize(config_.correspondence.target);
    }
    return correspondences;
}

std::pair<Eigen::Matrix<float, 6, 1>, float> OdometrySystem::solve(const SemanticCloud& points, const OdometrySystem::CorrespondenceSet& correspondences) noexcept {
    auto compute_jacobian_and_residual = [](const Eigen::Vector3f& source, const Eigen::Vector3f& target) {
        const Eigen::Vector3f residual = source - target;
        Eigen::Matrix<float, 3, 6> J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3f::hat(source);
        return std::make_tuple(J_r, residual);
    };

    using System = std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>, float, float>;
    auto sum = [](System a, const System &b) {
        auto& [a_H, a_b, a_chi, a_w] = a;
        const auto& [b_H, b_b, b_chi, b_w] = b;
        a_H += b_H;
        a_b += b_b;
        a_chi += b_chi;
        a_w += b_w;
        return a;
    };

    using correspondence_iterator = tbb::concurrent_vector<Correspondence>::const_iterator;
    const auto &[JTJ, JTr, chi, w] = tbb::parallel_reduce(
        tbb::blocked_range<correspondence_iterator>{correspondences.cbegin(), correspondences.cend()},
        System(Eigen::Matrix<float, 6, 6>::Zero(), Eigen::Matrix<float, 6, 1>::Zero(), 0.f, 0.f),
        [&](const tbb::blocked_range<correspondence_iterator>& range, System J) -> System {
            return std::transform_reduce(range.begin(), range.end(), J, sum, [&](const Correspondence& correspondence) {
                    const auto& [source, data, target] = correspondence;
                    const auto& [J_r, residual] = compute_jacobian_and_residual(source, target.point);
                    // weight the kernel based on confidence from the segmentation model
                    return System(J_r.transpose() * target.weight * J_r, J_r.transpose() * target.weight * residual, target.weight * residual.squaredNorm(), target.weight);
                });
        }, sum);

    return std::make_pair(JTJ.ldlt().solve(-JTr), w > 0.f ? chi / w : 0.f);
}

std::pair<Sophus::SE3f, Sophus::SE3f> OdometrySystem::align(SemanticCloud& points, const Sophus::SE3f& delta, Profiler& profiler) {
    // determine starting point
    Sophus::SE3f initial_guess = last_pose_ * delta;
    if(map_.empty()) return std::make_pair(initial_guess, Sophus::SE3f{});
    // compute adaptive threshold
    float sigma = threshold_.compute();
    float chi = -1.f;
    points.transform(initial_guess);
    {
        experimental.bev_curr.clear();
        points.project(experimental.bev_curr, experimental.ground);
        if(std::optional<Sophus::SE2f> T_bev_projected = align_projection(experimental.bev_curr, experimental.bev_prev, config_.max_iterations, config_.convergence_criterion, sigma * OdometrySystem::CORRESPONDENCE_MULTIPLIER)) {
            Sophus::SE3f T_bev = experimental.ground.lift(*T_bev_projected);
            CorrespondenceSet correspondences_initial = OdometrySystem::correspond(points, sigma * OdometrySystem::CORRESPONDENCE_MULTIPLIER);
            const auto [dx_initial, chi_initial] = OdometrySystem::solve(points, correspondences_initial);
            points.transform(T_bev);
            CorrespondenceSet correspondences_refined = OdometrySystem::correspond(points, sigma * OdometrySystem::CORRESPONDENCE_MULTIPLIER);
            const auto [dx_refined, chi_refined] = OdometrySystem::solve(points, correspondences_refined);
            if(chi_refined > chi_initial) {
                points.transform(T_bev.inverse());
            } else {
                initial_guess = T_bev * initial_guess;
            }
        }
    }
    // retrieve fields ahead of time to avoid repeated queries
    // perform ICP
    Sophus::SE3f T_icp{};
    PROFILE(align, &profiler) {
        for(size_t i = 0; i < config_.max_iterations; ++i) {
            // accumulate iteration count
            PROFILE_SKIP(align) statistics.iterations++;
            CorrespondenceSet correspondences; 
            PROFILE(iteration, &align) {
                // get correspondences
                correspondences = OdometrySystem::correspond(points, sigma * OdometrySystem::CORRESPONDENCE_MULTIPLIER);
                // update statistics
                PROFILE_SKIP(iteration) statistics.correspondences += static_cast<float>(correspondences.size());    
                // Equation (11)
                const auto [dx, chi_i] = OdometrySystem::solve(points, correspondences);
                const Sophus::SE3f est = Sophus::SE3f::exp(dx);
                // Equation (12)
                points.transform(est);
                // Update iterations
                T_icp = est * T_icp;
                // chi-based termination criteria
                if(chi > 0.f && std::abs(chi - chi_i) <= config_.convergence_criterion * chi) goto termination;
                chi = chi_i;
            }
            PROFILE_SKIP(align) {
                float threshold_min = std::numeric_limits<float>::max();
                float threshold_max = std::numeric_limits<float>::lowest();
                for(auto& correspondence : correspondences) {
                    const auto& [point, point_data, neighbor] = correspondence;
                    threshold_min = std::min(threshold_min, neighbor.weight);
                    threshold_max = std::max(threshold_max, neighbor.weight);
                }
                statistics.threshold.min += threshold_min;
                statistics.threshold.max += threshold_max;
            }
        }
termination:;
    }
    // compute new pose
    Sophus::SE3f pose = T_icp * initial_guess;
    PROFILE(experiment, &profiler) {
        experimental.ground = GroundPlane<SemanticCloud>(map_, pose.translation(), config_.max_distance_ground, 
            [](const Semantics& data) { return data.prediction == 1; });
        // std::vector<Eigen::Vector2f> projection = map_.transform_filter([this](const Eigen::Vector3f& point, const Semantics& semantics) -> std::optional<Eigen::Vector2f> {
        //     if(semantics.prediction != 3) return std::nullopt;
        //     return this->experimental.ground.project(point);
        // });
        experimental.bev_prev.clear();
        // experimental.bev_prev.extend(projection);
        points.project(experimental.bev_prev, experimental.ground);
    }
    // return transformation and deviation
    return std::make_pair(pose, initial_guess.inverse() * pose);
}

std::ostream& operator<<(std::ostream& stream, const OdometrySystem::Config& config) {
    auto format_threshold = [&stream](const std::map<size_t, float>& threshold) -> std::string {
        std::ostringstream oss;
        oss << '[';
        bool first = true;
        for(const auto [i, w] : threshold) {
            if(!first) oss << ", ";
            first = false; 
            if(i < classes::COUNT) {
                oss << classes::HINT[i];
            } else {
                oss << i;
            }
            oss << ": " << w;
        }
        oss << ']';
        return oss.str();
    };

    return stream << std::boolalpha << "OdometrySystem::Config" << std::endl <<
        "\tmax_distance: " << config.max_distance << "," << std::endl <<
        "\tmax_distance_ground: " << config.max_distance_ground << "," << std::endl <<
        "\tvoxel_size: " << config.voxel_size << "," << std::endl <<
        "\tvoxel_capacity: " << config.voxel_capacity << "," << std::endl <<
        "\tthreshold_initial: " << config.threshold.initial << "," << std::endl <<
        "\tthreshold_motion: " << config.threshold.motion << "," << std::endl <<
        "\tdeskew: " << config.deskew << "," << std::endl << 
        "\tfixed_dt: " << config.fixed_dt << "," << std::endl <<
        "\tmax_iterations: " << config.max_iterations << "," << std::endl <<
        "\tmax_threads: " << config.max_threads << "," << std::endl <<
        "\tconvergence_criterion: " << config.convergence_criterion << "," << std::endl <<
        "\tcorrespondence_weighting: " << config.correspondence.weighting << "," << std::endl <<
        "\tcorrespondence_target: " << config.correspondence.target;
}