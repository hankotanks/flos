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
#include "odometry/projection.h"
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

#if 0
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
#else
template<typename T>
void deskew_frame_inner(util::Cloud& points, const std::string& name, const Sophus::SE3f& motion, double dt) noexcept {
    const std::span<const T> stamps = points.field<T>(name);
    const auto [t_min_it, t_max_it] = std::minmax_element(stamps.begin(), stamps.end());
    const T t_min = *t_min_it;
    const T t_max = *t_max_it;
    const T dur = t_max - t_min;
    if(dur == T{0} || dt <= 1e-6) return;
    const float scale = static_cast<float>(static_cast<double>(dur) * 1e-9 / dt);

    const Eigen::Matrix<float, 6, 1> xi = scale * motion.log();
    const Eigen::Vector3f upsilon = xi.template head<3>();
    const Eigen::Vector3f omega = xi.template tail<3>();

    const float angular_speed = omega.norm();
    const float inverse_angular_speed = angular_speed > 1e-6f ? 1.f / angular_speed : 0.f;

    const Eigen::Vector3f axis = angular_speed > 1e-6f ? (omega * inverse_angular_speed).eval() : Eigen::Vector3f::Zero();

    const float duration_f = static_cast<float>(dur);

    tbb::parallel_for(tbb::blocked_range<size_t>{0, points.size()},
        [&](const tbb::blocked_range<size_t>& range) {
            for(size_t i = range.begin(); i < range.end(); ++i) {
                const float t = static_cast<float>(stamps[i] - t_min) / static_cast<float>(dur);
                const float t_neg = -t;

                const Eigen::Vector3f& point = points[i];
                if(angular_speed <= 1e-6f) {
                    points[i] = point + t_neg * upsilon;
                    continue;
                }

                const float theta = angular_speed * t_neg;
                const float sin_theta = std::sin(theta);
                const float cos_theta_neg = 1.f - std::cos(theta);

                const Eigen::Vector3f K_times_p = axis.cross(point);
                const Eigen::Vector3f R_times_p = point + sin_theta * K_times_p + cos_theta_neg * axis.cross(K_times_p);

                const Eigen::Vector3f K_times_upsilon = axis.cross(upsilon);
                const Eigen::Vector3f fst = (cos_theta_neg * inverse_angular_speed) * K_times_upsilon;
                const Eigen::Vector3f snd = ((theta - sin_theta) * inverse_angular_speed) * axis.cross(K_times_upsilon);;
                const Eigen::Vector3f V_times_rho = t_neg * upsilon + fst + snd;

                points[i] = R_times_p + V_times_rho;
            }
        });
}
#endif

void deskew_frame(util::Cloud& points, const Sophus::SE3f& motion, double dt) noexcept {
    auto name = get_stamp_field<float>(points);
    if(name) {
        deskew_frame_inner<float>(points, *name, motion, dt);
    }
    name = get_stamp_field<double>(points);
    if(name) deskew_frame_inner<double>(points, *name, motion, dt);
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
    projection_(config.max_distance, config_.max_distance_ground, config.voxel_size, config.voxel_capacity) {
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
        // deskewing
        if(config_.deskew) {
            PROFILE(deskew, &registration) {
                deskew_frame(points, last_delta_, last_stamp_ - prev_stamp_);
            }
        }
        // compute delta
        const auto& [delta, dt] = OdometrySystem::predict(seconds, nanoseconds);
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
            // refine initial_guess (invariant: point_kp_wrapper is transformed by initial_guess)
            if(config_.projection.enable) {
                PROFILE(refinement, &align) {
                    if(frame_ > config_.voxel_capacity) OdometrySystem::refine(points_kp_wrapper, initial_guess);
                }
            }   
            // ICP
            Sophus::SE3f T_icp = OdometrySystem::align(points_kp_wrapper, initial_guess);
            // build the current estimated pose
            Sophus::SE3f pose = T_icp * initial_guess;
            // update last pose
            last_delta_ = last_pose_.inverse() * pose;
            last_pose_ = pose;
            // maintain ground plane and previous frame's projection
            if(config_.projection.enable) {
                PROFILE(refinement_update, &align) {
                    if(frame_ > config_.voxel_capacity) {
                        projection_.fit(map_, last_pose_, T_icp, [this](const Semantics& data) { 
                            return this->config_.projection.ground.count(data.prediction) > 0; 
                        });
                    }
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

OdometrySystem::CorrespondenceSet OdometrySystem::correspond(const SemanticCloud& points) const noexcept {
    tbb::concurrent_vector<Correspondence> correspondences;
    correspondences.reserve(points.size());
    // collect all correspondences
    tbb::parallel_for(tbb::blocked_range<size_t>{0, points.size()}, [this, &points, &correspondences](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            const auto& [point, data] = points[i];
            const auto neighbor = this->map_.closest(point, data);
            if(neighbor) correspondences.emplace_back(point, data, *neighbor);
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

Sophus::SE2f OdometrySystem::refine(SemanticCloud& points, Sophus::SE3f& initial_guess) {
    BirdsEyeCloud bev{points, projection_.ground_, [this](const Semantics& data) {
        return this->config_.projection.target.count(data.prediction) > 0;
    }};
    if(std::optional<Sophus::SE2f> T_bev_projected = projection_.align(bev, config_.max_iterations, config_.convergence_criterion)) {
        Sophus::SE3f T_bev = projection_.ground_.lift(*T_bev_projected);
        // ensure that the refined residual is smaller than the initial
        CorrespondenceSet correspondences_initial = OdometrySystem::correspond(points);
        const auto [dx_initial, chi_initial] = OdometrySystem::solve(correspondences_initial);
        points.transform(T_bev);
        CorrespondenceSet correspondences_refined = OdometrySystem::correspond(points);
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

Sophus::SE3f OdometrySystem::align(SemanticCloud& points, const Sophus::SE3f& initial_guess) {
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
        correspondences = OdometrySystem::correspond(points);
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