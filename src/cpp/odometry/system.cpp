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
            points_kp.downsample(config_.voxel_size);
        }
        // compute pose w/ ICP
        Wrapper points_kp_wrapper{points_kp};
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
            Wrapper points_wrapper{points};
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

OdometrySystem::Belief::Belief() { 
    // populate will a flat distribution
    // NOTE: distribution will be populated by the VoxelMap::update, so we don't need to worry about it
    static constexpr float UNIFORM_CONFIDENCE = 1.f / static_cast<float>(classes::COUNT);
    confidence.fill(UNIFORM_CONFIDENCE);
    static const float UNIFORM_BELIEF = std::log(UNIFORM_CONFIDENCE);
    belief.fill(UNIFORM_BELIEF);
    prediction = std::numeric_limits<size_t>::max();
}

void OdometrySystem::Belief::update(const OdometrySystem::Semantics& data) noexcept {
    static constexpr float EPS = 1e-6f;
    for(size_t i = 0; i < classes::COUNT; ++i) {
        belief[i] += std::log(std::max(data.confidence[i], EPS));
    }
}

void OdometrySystem::Belief::finish() noexcept {
    auto it = std::max_element(belief.begin(), belief.end());
    // update predicted class of this voxel
    prediction = std::distance(belief.begin(), it);
    // renormalize belief so it doesn't keep growing
    float maxima = *it;
    for(auto& b : belief) b -= maxima;
    // compute confidence probabilities for future VoxelMap::closest queries
    for(size_t i = 0; i < classes::COUNT; ++i) confidence[i] = std::exp(belief[i]);
    float sum = confidence[0];
    for(size_t i = 1; i < classes::COUNT; ++i) sum += confidence[i];
    for(size_t i = 0; i < classes::COUNT; ++i) confidence[i] /= sum;
}

bool OdometrySystem::Belief::accept(const OdometrySystem::Semantics& query) const noexcept {
    return prediction == query.prediction;
}

#if 0
float OdometrySystem::Belief::weight(const OdometrySystem::Semantics& query) const noexcept {
    float jsd = 0.f;
    for(size_t i = 0; i < classes::COUNT; ++i) {
        const float p = confidence[i];
        const float q = query.confidence[i];
        const float m = 0.5f * (p + q);
        if(m == 0.f) continue;
        if(p > 0.f) jsd += 0.5f * p * std::log2(p / m);
        if(q > 0.f) jsd += 0.5f * q * std::log2(q / m);
    }
    return 1.f - std::sqrt(std::clamp(jsd, 0.f, 1.f));
}
#else
float OdometrySystem::Belief::weight(const OdometrySystem::Semantics& query) const noexcept {
    float w = 0.f;
    for(size_t i = 0; i < classes::COUNT; ++i) w += confidence[i] * query.confidence[i];
    assert(w >= 0.f && w <= 1.f);
    return w;
}
#endif

OdometrySystem::Wrapper::Wrapper(util::Cloud& points) : points_(points.points()), class_(points.field<uint8_t>("class")) {
    for(size_t i = 0; i < classes::COUNT; ++i) {
        confidence_[i] = points.field<float>(std::to_string(i));
    }
}

OdometrySystem::Semantics OdometrySystem::Wrapper::data(size_t i) const noexcept {
    Semantics data;
    data.prediction = class_[i];
    for(size_t j = 0; j < classes::COUNT; ++j) {
        data.confidence[j] = confidence_[j][i];
    }
    return data;
}

void OdometrySystem::Wrapper::transform(const Sophus::SE3f& t) const noexcept { 
    std::for_each(points_.begin(), points_.end(), [&t](Eigen::Vector3f& p) { p = t * p; }); 
}

std::pair<Eigen::Vector3f&, OdometrySystem::Semantics> OdometrySystem::Wrapper::operator[](size_t i) noexcept { 
    return std::pair<Eigen::Vector3f&, Semantics>(points_[i], Wrapper::data(i)); 
}

std::pair<const Eigen::Vector3f&, OdometrySystem::Semantics> OdometrySystem::Wrapper::operator[](size_t i) const noexcept { 
    return std::pair<const Eigen::Vector3f&, Semantics>(points_[i], Wrapper::data(i)); 
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

OdometrySystem::CorrespondenceSet OdometrySystem::correspond(const Wrapper& points, float correspondence_distance) const noexcept {
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

std::pair<Eigen::Matrix<float, 6, 1>, float> OdometrySystem::solve(const Wrapper& points, const OdometrySystem::CorrespondenceSet& correspondences) noexcept {
    auto compute_jacobian_and_residual = [](const Eigen::Vector3f& source, const Eigen::Vector3f& target) {
        const Eigen::Vector3f residual = source - target;
        Eigen::Matrix<float, 3, 6> J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3f::hat(source);
        return std::make_tuple(J_r, residual);
    };

    using System = std::tuple<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>, float>;
    auto sum = [](System a, const System &b) {
        auto& [a_H, a_b, a_chi] = a;
        const auto& [b_H, b_b, b_chi] = b;
        a_H += b_H;
        a_b += b_b;
        a_chi += b_chi;
        return a;
    };

    using correspondence_iterator = tbb::concurrent_vector<Correspondence>::const_iterator;
    const auto &[JTJ, JTr, chi] = tbb::parallel_reduce(
        tbb::blocked_range<correspondence_iterator>{correspondences.cbegin(), correspondences.cend()},
        System(Eigen::Matrix<float, 6, 6>::Zero(), Eigen::Matrix<float, 6, 1>::Zero(), 0.f),
        [&](const tbb::blocked_range<correspondence_iterator>& range, System J) -> System {
            return std::transform_reduce(range.begin(), range.end(), J, sum, [&](const Correspondence& correspondence) {
                    const auto& [source, data, target] = correspondence;
                    const auto& [J_r, residual] = compute_jacobian_and_residual(source, target.point);
                    // weight the kernel based on confidence from the segmentation model
                    return System(J_r.transpose() * target.weight * J_r, J_r.transpose() * target.weight * residual, residual.squaredNorm());
                });
        }, sum);

    return std::make_pair(JTJ.ldlt().solve(-JTr), chi);
}

std::pair<Sophus::SE3f, Sophus::SE3f> OdometrySystem::align(Wrapper& points, const Sophus::SE3f& delta, Profiler& profiler) {
    // determine starting point
    Sophus::SE3f initial_guess = last_pose_ * delta;
    if(map_.empty()) return std::make_pair(initial_guess, Sophus::SE3f{});
    // compute adaptive threshold
    float sigma = threshold_.compute();
    float chi = -1.f;
    points.transform(initial_guess);
    {
        experimental.bev_prev = std::move(experimental.bev_prev_next);
        experimental.bev_curr = points.project(experimental.centroid, experimental.plane, config_.max_distance, config_.voxel_size, config_.voxel_capacity);
        experimental.accepted = false;
        experimental.bev_curr_post = experimental.bev_curr;
        if(std::optional<Sophus::SE2f> T_bev_projected = experimental.bev_curr.align(experimental.bev_prev, config_.max_iterations, config_.convergence_criterion, sigma * OdometrySystem::CORRESPONDENCE_MULTIPLIER)) {
            Sophus::SE3f T_bev = OdometrySystem::lift(*T_bev_projected, experimental.centroid, experimental.plane);
            CorrespondenceSet correspondences_initial = OdometrySystem::correspond(points, sigma * OdometrySystem::CORRESPONDENCE_MULTIPLIER);
            const auto [dx_initial, chi_initial] = OdometrySystem::solve(points, correspondences_initial);
            points.transform(T_bev);
            CorrespondenceSet correspondences_refined = OdometrySystem::correspond(points, sigma * OdometrySystem::CORRESPONDENCE_MULTIPLIER);
            const auto [dx_refined, chi_refined] = OdometrySystem::solve(points, correspondences_refined);
            if(chi_refined > chi_initial) {
                points.transform(T_bev.inverse());
            } else {
                initial_guess = T_bev * initial_guess;
                experimental.accepted = true;
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
        const auto [centroid, plane] = OdometrySystem::fit_ground_plane(pose.translation());
        experimental.centroid = centroid;
        experimental.plane = plane;
        experimental.bev_prev_next = points.project(experimental.centroid, experimental.plane, config_.max_distance, config_.voxel_size, config_.voxel_capacity);
    }
    // return transformation and deviation
    return std::make_pair(pose, initial_guess.inverse() * pose);
}

std::pair<Eigen::Vector3f, Eigen::Matrix3f> OdometrySystem::fit_ground_plane(const Eigen::Vector3f& origin) noexcept {
    // centroid
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    size_t count{0};
    map_.for_each_in_radius(origin, config_.max_distance_fit, [&centroid, &count](const Eigen::Vector3f& point, const OdometrySystem::Semantics& semantics) {
        if(semantics.prediction != 1) return;
        centroid += point;
        count++;
    });
    centroid /= static_cast<float>(count);
    // covariance
    Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
    map_.for_each_in_radius(origin, config_.max_distance_fit, [&centroid, &S, &count](const Eigen::Vector3f& point, const OdometrySystem::Semantics& semantics) {
        if(semantics.prediction != 1) return;
        const Eigen::Vector3f d = point - centroid;
        S += d * d.transpose();
    });
    S /= static_cast<float>(count);
    // solve
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver{S};
    Eigen::Matrix3f plane = solver.eigenvectors();
    if(plane.col(1).cross(plane.col(2)).dot(plane.col(0)) < 0.f) plane.col(2) *= -1.f;
    return std::make_pair(centroid, plane);
}

Sophus::SE3f OdometrySystem::lift(const Sophus::SE2f& dT, const Eigen::Vector3f& centroid, const Eigen::Matrix3f& plane) {
    Eigen::Matrix3f R_plane = Eigen::Matrix3f::Identity();
    R_plane.block<2, 2>(1, 1) = dT.so2().matrix();

    const Eigen::Matrix3f R3 =
        plane * R_plane * plane.transpose();

    const Eigen::Vector3f t3 =
        centroid
        + plane.block<3, 2>(0, 1) * dT.translation()
        - R3 * centroid;

    return Sophus::SE3f{R3, t3};
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
        "\tcorrespondence_target: " << config.correspondence.target;
}