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
} // private scope

namespace {
bool pred_point(const Belief& voxel, const Semantics& point, const Semantics& query) {
    (void) voxel;
    return query.prediction == point.prediction;
}

bool pred_voxel(const Belief& voxel, const Semantics& point, const Semantics& query) {
    (void) point;
    return query.prediction == voxel.prediction;
}

bool pred_const(const Belief& voxel, const Semantics& point, const Semantics& query) {
    (void) voxel;
    (void) point;
    (void) query;
    return true;
}
} // private scope

namespace {
float weight_point(const VoxelMap<Wrapper>::Neighbor& neighbor, const Semantics& query) {
    float w = 0.f;
    // NOTE: currently, dot product
    // consider KL or Jensen-Shannon
    for(size_t i = 0; i < classes::COUNT; ++i) w += neighbor.data.confidence[i] * query.confidence[i];
    assert(w >= 0.f && w <= 1.f);
    return w;
}

float weight_voxel(const VoxelMap<Wrapper>::Neighbor& neighbor, const Semantics& query) {
    (void) query;
    return neighbor.weight;
}

float weight_const(const VoxelMap<Wrapper>::Neighbor& neighbor, const Semantics& query) {
    (void) neighbor;
    (void) query;
    return 1.f;
}
} // private scope

namespace {
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
    pred_(config.correspondence.restrict ? (config.correspondence.weighting_by_belief ? pred_voxel : pred_point) : pred_const),
    weight_(config.correspondence.weighting ? (config.correspondence.weighting_by_belief ? weight_voxel : weight_point) : weight_const),
    map_(config.max_distance, config.voxel_size, config.voxel_capacity),
    threshold_(config.max_distance, config.threshold.initial, config.threshold.motion) {
    size_t max_threads = config.max_threads > 0 ? \
        config.max_threads : tbb::this_task_arena::max_concurrency();
    volatile static const tbb::global_control _ = tbb::global_control( // NOTE: static storage type is required here
        tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_threads));
    // create lookup table for min_weight
    correspondence_threshold_.resize(classes::COUNT);
    for(const auto& [cls, weight] : config_.correspondence.threshold) {
        correspondence_threshold_[cls] = weight;
    }   
}

const Sophus::SE3f& OdometrySystem::register_to_map(util::Cloud& points) noexcept {
    config_.fixed_dt = true;
    return OdometrySystem::register_to_map(points, 0, 0);
}

const Sophus::SE3f& OdometrySystem::register_to_map(util::Cloud& points, unsigned int seconds, unsigned int nanoseconds) noexcept {
    PROFILE(registration, this) {
        // compute delta
        Sophus::SE3f delta = OdometrySystem::prologue(seconds, nanoseconds);
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
        Wrapper points_kp_wrapper{points_kp};
        const auto [pose, deviation] = OdometrySystem::align(points_kp_wrapper, delta);
        // update AdaptiveThreshold
        threshold_.update(deviation);
        // append points to map
        PROFILE(update, &registration) {
            Wrapper points_wrapper{points};
            map_.update(points_wrapper, pose);
        }
        // store new pose
        OdometrySystem::epilogue(pose, seconds, nanoseconds);
    }
    return last_pose_;
}

Sophus::SE3f OdometrySystem::prologue(unsigned int seconds, unsigned int nanoseconds) noexcept {
    // reset statistics
    statistics = Statistics{};
    // handling depends on whether we rely on timestamps
    if(config_.fixed_dt) {
        return last_delta_;
    } else {
        if(init_seconds_ == std::numeric_limits<unsigned int>::max()) init_seconds_ = seconds;
        float stamp = static_cast<float>(seconds - init_seconds_) + 1e-9 * static_cast<float>(nanoseconds);
        float dt_prev = last_stamp_ - prev_stamp_;
        float dt_curr = stamp - last_stamp_;
        float scale = 1.f;
        if(dt_prev > 1e-6) scale = static_cast<float>(dt_curr / dt_prev);
        // compute predicted delta:
        return Sophus::SE3f::exp(scale * last_delta_.log());
    }
}

void OdometrySystem::epilogue(const Sophus::SE3f& pose, unsigned int seconds, unsigned int nanoseconds) noexcept {
    // update last pose
    last_delta_ = last_pose_.inverse() * pose;
    last_pose_ = pose;
    // if we aren't handling variable dt, theres no need to update stamps or velocity statistics
    if(!config_.fixed_dt) {
        float stamp = static_cast<float>(seconds - init_seconds_) + 1e-9 * static_cast<float>(nanoseconds);
        float dt_curr = stamp - last_stamp_;
        // update timestamps
        prev_stamp_ = last_stamp_;
        last_stamp_ = stamp;
        // log final velocity statistics
        Eigen::Matrix<float, 6, 1> xi = last_delta_.log();
        statistics.velocity.linear = linear_velocity(xi, dt_curr);
        statistics.velocity.angular = angular_velocity(xi, dt_curr);
    }
}

OdometrySystem::CorrespondenceSet OdometrySystem::correspond(const Wrapper& points, float sigma) const noexcept {
    tbb::concurrent_vector<Correspondence> correspondences;
    correspondences.reserve(points.size());
    float correspondence_distance_sq = sigma * sigma * OdometrySystem::CORRESPONDENCE_MULTIPLIER * OdometrySystem::CORRESPONDENCE_MULTIPLIER;
    tbb::parallel_for(tbb::blocked_range<size_t>{0, points.size()}, [this, &points, &correspondences, correspondence_distance_sq](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            const auto& [point, point_data] = points[i];
            const auto& maybe = this->map_.closest(point, point_data, pred_, correspondence_distance_sq);
            if(maybe) {
                const auto& neighbor = *maybe;
                if(this->weight_(neighbor, point_data) < correspondence_threshold_[point_data.prediction]) continue;
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
    auto sum = [](System a, const System &b) {
        a.first += b.first;
        a.second += b.second;
        return a;
    };

    using correspondence_iterator = tbb::concurrent_vector<Correspondence>::const_iterator;
    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        tbb::blocked_range<correspondence_iterator>{correspondences.cbegin(), correspondences.cend()},
        System(Eigen::Matrix<float, 6, 6>::Zero(), Eigen::Matrix<float, 6, 1>::Zero()),
        [&](const tbb::blocked_range<correspondence_iterator>& range, System J) -> System {
            return std::transform_reduce(range.begin(), range.end(), J, sum, [&](const Correspondence& correspondence) {
                    const auto& [source, data, target] = correspondence;
                    const auto& [J_r, residual] = compute_jacobian_and_residual(source, target.point);
                    // weight the kernel based on confidence from the segmentation model
                    float w = config_.correspondence.weighting ? target.weight : 1.f;
                    return System(J_r.transpose() * w * J_r, J_r.transpose() * w * residual);
                });
        }, sum);

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
        for(size_t j = 0; j < config_.max_iterations; ++j) {
            PROFILE(iteration, &align) {
                // get correspondences
                CorrespondenceSet correspondences = OdometrySystem::correspond(points, sigma);
                // update statistics
                statistics.correspondences += static_cast<float>(correspondences.size());
                // Equation (11)
                const Eigen::Matrix<float, 6, 1> dx = OdometrySystem::solve(points, correspondences);
                const Sophus::SE3f est = Sophus::SE3f::exp(dx);
                // Equation (12)
                points.transform(est);
                // Update iterations
                T_icp = est * T_icp;
                // Termination criteria
                if(dx.norm() < config_.convergence_criterion) goto termination;
            }            
            // accumulate iteration count
            statistics.iterations++;
        }
termination:;
    }
    // get mean correspondences
    statistics.correspondences /= static_cast<float>(statistics.iterations);
    // compute new pose
    Sophus::SE3f pose = T_icp * initial_guess;
    // return transformation and deviation
    return std::make_pair(pose, initial_guess.inverse() * pose);
}

void OdometrySystem::AdaptiveThreshold::update(const Sophus::SE3f& deviation) noexcept {
    const float model_error = [&]() {
        const float theta = Eigen::AngleAxisf(deviation.rotationMatrix()).angle();
        const float delta_rot = 2.0 * max_distance_ * std::sin(theta / 2.0);
        const float delta_trans = deviation.translation().norm();
        return delta_trans + delta_rot;
    }();
    if (model_error > threshold_motion_) {
        sse_ += model_error * model_error;
        samples_++;
    }
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
        "\tcorrespondence_weighting_by_belief: " << config.correspondence.weighting_by_belief << "," << std::endl <<
        "\tcorrespondence_restrict: " << config.correspondence.restrict << "," << std::endl <<
        "\tcorrespondence_threshold: " << format_threshold(config.correspondence.threshold);
}