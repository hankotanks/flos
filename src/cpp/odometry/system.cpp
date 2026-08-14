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
    projection_(config.max_distance, config.voxel_size, config.voxel_capacity) {
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
    statistics_ = Statistics{};
    // perform registration
    PROFILE(registration, this) {
        // compute delta
        const auto& [delta, dt] = OdometrySystem::predict(seconds, nanoseconds);
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
        // get keypoints
        SemanticCloud points_kp_wrapper{points_kp};
        // prepare rough initial guess
        Sophus::SE3f initial_guess = last_pose_ * delta;
        points_kp_wrapper.transform(initial_guess);
        // initial_guess refinement
        PROFILE(align, &registration) {
            // compute adaptive threshold
            float sigma = threshold_.compute();
            // and correspondence distance
            float correspondence_distance = sigma * OdometrySystem::CORRESPONDENCE_MULTIPLIER;
            if(config_.projection.enable) {
                PROFILE(refinement, &align) {
                    if(frame_ > config_.voxel_capacity) {
                        OdometrySystem::project(projection_.curr, points_kp_wrapper);
                        OdometrySystem::refine(points_kp_wrapper, initial_guess, correspondence_distance);
                    }
                }
            }   
            // ICP
            Sophus::SE3f T_icp = OdometrySystem::align(points_kp_wrapper, initial_guess, sigma);
            // build the current estimated pose
            Sophus::SE3f pose = T_icp * initial_guess;
            // update AdaptiveThreshold with deviation
            threshold_.update(initial_guess.inverse() * pose);
            // update last pose
            last_delta_ = last_pose_.inverse() * pose;
            last_pose_ = pose;
            // maintain ground plane and previous frame's projection
            if(config_.projection.enable) {
                PROFILE(refinement_update, &align) {
                    if(frame_ > config_.voxel_capacity) OdometrySystem::fit(points_kp_wrapper, T_icp);
                }
            }
        }
        // append points to map
        PROFILE(update, &registration) {
            SemanticCloud points_wrapper{points};
            map_.update(points_wrapper, last_pose_);
        }
        // if we aren't handling variable dt, theres no need to update stamps or velocity statistics
        PROFILE_SKIP(registration) {
            // log final velocity statistics
            Eigen::Matrix<float, 6, 1> xi = last_delta_.log();
            statistics_.velocity.linear = linear_velocity(xi, dt);
            statistics_.velocity.angular = angular_velocity(xi, dt);
        }
    }
    ++frame_;
    // return const ref to the computed pose, since we store it anyway
    return last_pose_;
}

std::pair<Sophus::SE3f, double> OdometrySystem::predict(unsigned int seconds, unsigned int nanoseconds) noexcept {
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

void OdometrySystem::fit(const SemanticCloud& points, const Sophus::SE3f& T) {
    if((projection_.origin - last_pose_.translation()).squaredNorm() > (config_.max_distance_ground * config_.max_distance_ground * 0.25f)) {
        projection_.origin = last_pose_.translation();
        projection_.ground = GroundPlane<SemanticCloud>(map_, projection_.origin, config_.max_distance_ground, 
            [this](const Semantics& data) { return this->config_.projection.ground.count(data.prediction) > 0; });
        OdometrySystem::project(projection_.prev, points);
    } else {
        // update current frame projection
        projection_.prev = std::move(projection_.curr);
        Sophus::SE2f T_icp_projected = projection_.ground.project(T);
        projection_.prev.apply(T_icp_projected);
    }
}

void OdometrySystem::project(VoxelMap<2>& surface, const SemanticCloud& points) {
    surface.clear();
    for(size_t i = 0; i < points.size(); ++i) {
        const auto [point, semantics] = points[i];
        if(config_.projection.target.count(semantics.prediction) == 0) continue;
        surface.extend(projection_.ground.project(point));
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

std::pair<Eigen::Matrix<float, 6, 1>, float> OdometrySystem::solve(const OdometrySystem::CorrespondenceSet& correspondences) noexcept {
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

Sophus::SE3f OdometrySystem::align(SemanticCloud& points, const Sophus::SE3f& initial_guess, float sigma) {
    // determine starting point
    if(map_.empty()) return {};
    // termination threshold (Ceres-style)
    float chi = -1.f;
    // perform ICP
    Sophus::SE3f T_icp{};
    for(size_t i = 0; i < config_.max_iterations; ++i) {
        // accumulate iteration count
        statistics_.iterations++;
        CorrespondenceSet correspondences; 
        // get correspondences
        correspondences = OdometrySystem::correspond(points, sigma * OdometrySystem::CORRESPONDENCE_MULTIPLIER);
        // update statistics
        statistics_.correspondences += static_cast<float>(correspondences.size());    
        // Equation (11)
        const auto [dx, chi_i] = OdometrySystem::solve(correspondences);
        const Sophus::SE3f est = Sophus::SE3f::exp(dx);
        // Equation (12)
        points.transform(est);
        // Update iterations
        T_icp = est * T_icp;
        // chi-based termination criteria
        if(chi > 0.f && std::abs(chi - chi_i) <= config_.convergence_criterion * chi) break;
        chi = chi_i;
    }
    statistics_.correspondences /= static_cast<float>(statistics_.iterations);
    // return transformation and deviation
    return T_icp;
}

Sophus::SE2f OdometrySystem::refine(SemanticCloud& points, Sophus::SE3f& initial_guess, float correspondence_distance) {
    if(std::optional<Sophus::SE2f> T_bev_projected = projection_.align(config_.max_iterations, config_.convergence_criterion, correspondence_distance)) {
        Sophus::SE3f T_bev = projection_.ground.lift(*T_bev_projected);
        // ensure that the refined residual is smaller than the initial
        CorrespondenceSet correspondences_initial = OdometrySystem::correspond(points, correspondence_distance);
        const auto [dx_initial, chi_initial] = OdometrySystem::solve(correspondences_initial);
        points.transform(T_bev);
        CorrespondenceSet correspondences_refined = OdometrySystem::correspond(points, correspondence_distance);
        const auto [dx_refined, chi_refined] = OdometrySystem::solve(correspondences_refined);
        // we are essentially sacrificing 2 icp iterations under the assumption that the refinement will reduce the total ICP count by more than 2
        if(chi_refined > chi_initial) {
            points.transform(T_bev.inverse());
            return {};
        } else {
            initial_guess = T_bev * initial_guess;
            return *T_bev_projected;
        }
    }
    return {};
}

std::ostream& operator<<(std::ostream& stream, const OdometrySystem::Config& config) {
    auto format = [&stream](const std::set<size_t>& classes) -> std::string {
        std::ostringstream oss;
        oss << '[';
        bool first = true;
        for(const auto i : classes) {
            if(!first) oss << ", ";
            first = false; 
            if(i < classes::COUNT) {
                oss << classes::HINT[i];
            } else {
                oss << i;
            }
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
        "\tcorrespondence_target: " << config.correspondence.target << "," << std::endl <<
        "\tprojection_enable: " << config.projection.enable << "," << std::endl <<
        "\tprojection_ground: " << format(config.projection.ground) << "," << std::endl <<
        "\tprojection_target: " << format(config.projection.target);
}