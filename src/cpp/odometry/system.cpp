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
#include <tbb/enumerable_thread_specific.h>
// package
#include "classes.h"
// util
#include "odometry/wrapper.h"
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

bool pred_query_to_point(const Belief& data, Semantics point, Semantics query) {
    return query.prediction == point.prediction;
}

bool pred_query_to_voxel(const Belief& data, Semantics point, Semantics query) {
    return query.prediction == data.prediction;
}

bool pred_none(const Belief& data, Semantics point, Semantics query) {
    return true;
}

float linear_velocity(const Eigen::Matrix<float,6,1>& xi, float dt) {
    return xi.tail<3>().norm() / dt;
}

Eigen::Vector3f angular_wrap(const Eigen::Vector3f& rot) {
    double angle = rot.norm();
    if(angle < 1e-12) return rot;
    double wrapped = std::fmod(angle + M_PI, 2.0 * M_PI) - M_PI;
    return rot * (wrapped / angle);
}

float angular_velocity(const Eigen::Matrix<float,6,1>& xi, float dt) {
    Eigen::Vector3f omega = angular_wrap(xi.head<3>()) / dt;
    return omega.norm();
}
} // private scope

OdometrySystem::OdometrySystem(const Config& config) : util::Profiler("OdometrySystem"),
    config_(config),
    pred_(config.registration.correspondence_restriction ? (config.registration.correspondence_with_voxel_belief ? pred_query_to_voxel : pred_query_to_point) : pred_none),
    map_(config.map.max_distance, config.map.voxel_size, config.map.voxel_capacity),
    threshold_(config.map.max_distance, config.threshold.threshold_initial, config.threshold.min_motion) {
    size_t max_threads = config.registration.max_threads > 0 ? \
        config.registration.max_threads : tbb::this_task_arena::max_concurrency();
    volatile static const tbb::global_control _ = tbb::global_control( // NOTE: static storage type is required here
        tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_threads));
    // create lookup table for min_weight
    if(!config_.map.min_weight.empty()) {
        min_weight_.resize(classes::COUNT);
        for(const auto& [cls, weight] : config_.map.min_weight) {
            min_weight_[cls] = weight;
        }
    }
}

const Sophus::SE3f& OdometrySystem::register_to_map(util::Cloud& points, unsigned int seconds, unsigned int nanoseconds) noexcept {
    PROFILE(registration, this) {
        // compute delta
        Sophus::SE3f delta;
        OdometrySystem::prologue(delta, seconds, nanoseconds);
        // deskewing
        if(config_.registration.deskew) {
            PROFILE(deskew, &registration) {
                deskew_frame(points, delta);
            }
        }
        // downsample incoming points to obtain keypoints and the preprocessed frame
        util::Cloud points_kp{points};
        PROFILE(downsample, &registration) {
            points_kp.downsample(config_.map.voxel_size * OdometrySystem::VOXEL_SIZE_SCALAR);
        }
        // compute pose w/ ICP
        Wrapper points_kp_wrapper{points_kp};
        const auto [pose, deviation] = OdometrySystem::align(points_kp_wrapper, delta);
        // update AdaptiveThreshold
        threshold_.update(deviation);
        // append points to map
        PROFILE(update, &registration) {
            Wrapper points_wrapper{points};
            map_.update([this](const Belief& belief, const Semantics& data) {
                return belief.weight(data) > min_weight_[data.prediction]; // TODO: Consider a configuration option for this
            }, points_wrapper, pose);
        }
        // store new pose
        OdometrySystem::epilogue(pose, seconds, nanoseconds);
    }
    return last_pose_;
}

void OdometrySystem::AdaptiveThreshold::update(const Sophus::SE3f& deviation) noexcept {
    const float model_error = [&]() {
        const float theta = Eigen::AngleAxisf(deviation.rotationMatrix()).angle();
        const float delta_rot = 2.0 * max_distance_ * std::sin(theta / 2.0);
        const float delta_trans = deviation.translation().norm();
        return delta_trans + delta_rot;
    }();
    if (model_error > min_motion_) {
        model_sse_ += model_error * model_error;
        num_sample_++;
    }
}

void OdometrySystem::prologue(Sophus::SE3f& delta, unsigned int seconds, unsigned int nanoseconds) noexcept {
    // reset statistics
    statistics = Statistics{};
    // handling depends on whether we rely on timestamps
    if(config_.registration.constant_velocity) {
        delta = last_delta_;
    } else {
        if(init_seconds_ == std::numeric_limits<unsigned int>::max()) init_seconds_ = seconds;
        float stamp = static_cast<float>(seconds - init_seconds_) + 1e-9 * static_cast<float>(nanoseconds);
        float dt_prev = last_stamp_ - prev_stamp_;
        float dt_curr = stamp - last_stamp_;
        float scale = 1.f;
        if(dt_prev > 1e-6) scale = static_cast<float>(dt_curr / dt_prev);
        // compute predicted delta:
        delta = Sophus::SE3f::exp(scale * last_delta_.log());
        // log initial velocity statistics
        Eigen::Matrix<float, 6, 1> xi = delta.log();
        statistics.velocity.linear.initial = linear_velocity(xi, dt_curr);
        statistics.velocity.angular.initial = angular_velocity(xi, dt_curr);
    }
}

void OdometrySystem::epilogue(const Sophus::SE3f& pose, unsigned int seconds, unsigned int nanoseconds) noexcept {
    // update last pose
    last_delta_ = last_pose_.inverse() * pose;
    last_pose_ = pose;
    // if we aren't handling variable dt, theres no need to update stamps or velocity statistics
    if(!config_.registration.constant_velocity) {
        float stamp = static_cast<float>(seconds - init_seconds_) + 1e-9 * static_cast<float>(nanoseconds);
        float dt_curr = stamp - last_stamp_;
        // update timestamps
        prev_stamp_ = last_stamp_;
        last_stamp_ = stamp;
        // log final velocity statistics
        Eigen::Matrix<float, 6, 1> xi = last_delta_.log();
        statistics.velocity.linear.solved = linear_velocity(xi, dt_curr);
        statistics.velocity.angular.solved = angular_velocity(xi, dt_curr);
    }
}

OdometrySystem::CorrespondenceSet OdometrySystem::correspond(const Wrapper& points, float sigma) const noexcept {
    tbb::concurrent_vector<Correspondence> correspondences;
    correspondences.reserve(points.size());
    float correspondence_distance_sq = sigma * sigma * config_.registration.correspondence_multiplier * config_.registration.correspondence_multiplier;
    tbb::parallel_for(tbb::blocked_range<size_t>{0, points.size()}, [&](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            const auto& [point, point_data] = points[i];
            const auto& maybe = map_.closest(pred_, point, point_data, correspondence_distance_sq);
            if(maybe) {
                // TODO: It doesn't make sense to require data.prediction to match in the predicate,
                // then separately filter the correspondences based on the min_weight threshold
                // this should be unified in a single stage (once I've removed the point_to_point predicate query, which is outdated)
                const auto& neighbor = *maybe;
                if(neighbor.weight < min_weight_[neighbor.data.prediction]) continue;
                correspondences.emplace_back(point, point_data, neighbor);
            }
        }
    });
    return correspondences;
}

Eigen::Matrix<float, 6, 1> OdometrySystem::solve(const Wrapper& points, const OdometrySystem::CorrespondenceSet& correspondences) noexcept {
    auto compute_jacobian_and_residual = [](const Eigen::Vector3f& source, const Eigen::Vector3f& target) {
        const Eigen::Vector3f residual = source - target;
        Eigen::Matrix<float, 3, 6> J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3f::hat(source);
        return std::make_tuple(J_r, residual);
    };

    using System = std::pair<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>>;
    auto system_sum = [](System a, const System &b) {
        a.first += b.first;
        a.second += b.second;
        return a;
    };

    using correspondence_iterator = tbb::concurrent_vector<Correspondence>::const_iterator;
    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        tbb::blocked_range<correspondence_iterator>{correspondences.cbegin(), correspondences.cend()},
        System(Eigen::Matrix<float, 6, 6>::Zero(), Eigen::Matrix<float, 6, 1>::Zero()),
        [&](const tbb::blocked_range<correspondence_iterator>& range, System J) -> System {
            return std::transform_reduce(range.begin(), range.end(), J, system_sum, [&](const Correspondence& correspondence) {
                    const auto& [source, data, target] = correspondence;
                    const auto& [J_r, residual] = compute_jacobian_and_residual(source, target.point);
                    // weight the kernel based on confidence from the segmentation model
                    float w = config_.registration.confidence_weighting ? target.weight : 1.f;
                    return System(J_r.transpose() * w * J_r, J_r.transpose() * w * residual);
                });
        }, system_sum);

    return JTJ.ldlt().solve(-JTr);
}

std::pair<Sophus::SE3f, Sophus::SE3f> OdometrySystem::align(Wrapper& points, const Sophus::SE3f& delta) {
    // determine starting point
    Sophus::SE3f initial_guess = last_pose_ * delta;
    if(map_.empty()) return std::make_pair(initial_guess, Sophus::SE3f{});
    points.transform(initial_guess);
    // compute adaptive threshold
    float sigma = threshold_.compute();
    // retrieve fields ahead of time to avoid repeated queries
    // perform ICP
    Sophus::SE3f T_icp{};
    PROFILE(align, this) {
        for(size_t j = 0; j < config_.registration.max_iterations; ++j) {
            PROFILE(iteration, &align) {
                // get correspondences
                CorrespondenceSet correspondences = OdometrySystem::correspond(points, sigma);
                // update statistics
                if(statistics.iterations == 0) statistics.correspondence.initial_count = correspondences.size();
                statistics.correspondence.final_count = correspondences.size();
                statistics.correspondence.mean += static_cast<float>(statistics.correspondence.final_count);
                // Equation (11)
                const Eigen::Matrix<float, 6, 1> dx = OdometrySystem::solve(points, correspondences);
                const Sophus::SE3f est = Sophus::SE3f::exp(dx);
                // Equation (12)
                points.transform(est);
                // Update iterations
                T_icp = est * T_icp;
                // Termination criteria
                if(dx.norm() < config_.registration.convergence_criterion) goto termination;
            }            
            // accumulate iteration count
            statistics.iterations++;
        }
termination:;
    }
    // get mean correspondences
    statistics.correspondence.mean /= static_cast<float>(statistics.iterations);
    // compute new pose
    Sophus::SE3f pose = T_icp * initial_guess;
    // return transformation and deviation
    return std::make_pair(pose, initial_guess.inverse() * pose);
}

std::ostream& operator<<(std::ostream& stream, const OdometrySystem::Config& config) {
    auto format = [&stream](const std::map<size_t, float>& indices_to_weights) -> std::string {
        std::ostringstream oss;
        oss << '[';
        bool first = true;
        for(const auto [i, w] : indices_to_weights) {
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
        "\tmax_distance: " << config.map.max_distance << "," << std::endl <<
        "\tmin_weight: " << format(config.map.min_weight) << "," << std::endl <<
        "\tvoxel_size: " << config.map.voxel_size << "," << std::endl <<
        "\tvoxel_capacity: " << config.map.voxel_capacity << "," << std::endl <<
        "\tthreshold_initial: " << config.threshold.threshold_initial << "," << std::endl <<
        "\tmin_motion: " << config.threshold.min_motion << "," << std::endl <<
        "\tdeskew: " << config.registration.deskew << "," << std::endl << 
        "\tconfidence_weighting: " << config.registration.confidence_weighting << "," << std::endl <<
        "\tcorrespondence_restriction: " << config.registration.correspondence_restriction << "," << std::endl <<
        "\tcorrespondence_with_voxel_belief: " << config.registration.correspondence_with_voxel_belief << "," << std::endl <<
        "\tconstant_velocity: " << config.registration.constant_velocity << "," << std::endl <<
        "\tmax_iterations: " << config.registration.max_iterations << "," << std::endl <<
        "\tmax_threads: " << config.registration.max_threads << "," << std::endl <<
        "\tconvergence_criterion: " << config.registration.convergence_criterion << "," << std::endl <<
        "\tcorrespondence_multiplier: " << config.registration.correspondence_multiplier;
}