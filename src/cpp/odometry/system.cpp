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
#if 1
inline float weight(const std::array<float, classes::COUNT>& fst, const std::array<float, classes::COUNT>& snd) {
    float jsd = 0.f;
    for(size_t i = 0; i < classes::COUNT; ++i) {
        const float p = fst[i];
        const float q = snd[i];
        const float m = 0.5f * (p + q);
        if(m == 0.f) continue;
        if(p > 0.f) jsd += 0.5f * p * std::log2(p / m);
        if(q > 0.f) jsd += 0.5f * q * std::log2(q / m);
    }
    assert(jsd >= 0.f && jsd <= 1.f);
    return 1.f - std::sqrt(jsd);
}
#else
inline float weight(const std::array<float, classes::COUNT>& fst, const std::array<float, classes::COUNT>& snd) {
    float w = 0.f;
    for(size_t i = 0; i < classes::COUNT; ++i) w += fst[i] * snd[i];
    assert(w >= 0.f && w <= 1.f);
    return w;
}
#endif

inline float weight_point(const OdometrySystem::Belief& voxel, const OdometrySystem::Semantics& point, const OdometrySystem::Semantics& query) {
    (void) voxel;
    return weight(point.confidence, query.confidence);
}

inline float weight_voxel(const OdometrySystem::Belief& voxel, const OdometrySystem::Semantics& point, const OdometrySystem::Semantics& query) {
    (void) point;
    return weight(voxel.confidence, query.confidence);
}

inline float weight_const(const OdometrySystem::Belief& voxel, const OdometrySystem::Semantics& point, const OdometrySystem::Semantics& query) {
    (void) voxel;
    (void) point;
    (void) query;
    return 1.f;
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
    map_(config.max_distance, config.voxel_size, config.voxel_capacity, config.correspondence.weighting ? (config.correspondence.weighting_by_belief ? weight_voxel : weight_point) : weight_const),
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

OdometrySystem::Belief::Belief(const OdometrySystem::Weight& weight) : weight_(weight) { 
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

std::pair<Sophus::SE3f, float> OdometrySystem::predict_delta(unsigned int seconds, unsigned int nanoseconds) noexcept {
    // handling depends on whether we rely on timestamps
    if(config_.fixed_dt) {
        return std::make_pair(last_delta_, 1.f);
    } else {
        if(init_seconds_ == std::numeric_limits<unsigned int>::max()) init_seconds_ = seconds;
        float stamp = static_cast<float>(seconds - init_seconds_) + 1e-9 * static_cast<float>(nanoseconds);
        float dt_prev = last_stamp_ - prev_stamp_;
        float dt_curr = stamp - last_stamp_;
        float scale = 1.f;
        if(dt_prev > 1e-6) scale = static_cast<float>(dt_curr / dt_prev);
        // update timestamp state
        prev_stamp_ = last_stamp_;
        last_stamp_ = stamp;
        // compute predicted delta:
        return std::make_pair(Sophus::SE3f::exp(scale * last_delta_.log()), dt_curr);
    }
}

OdometrySystem::CorrespondenceSet OdometrySystem::correspond(const Wrapper& points, float sigma) const noexcept {
    tbb::concurrent_vector<Correspondence> correspondences;
    correspondences.reserve(points.size());
    // collect all correspondences
    float correspondence_distance_sq = sigma * sigma * OdometrySystem::CORRESPONDENCE_MULTIPLIER * OdometrySystem::CORRESPONDENCE_MULTIPLIER;
    tbb::parallel_for(tbb::blocked_range<size_t>{0, points.size()}, [this, &points, &correspondences, correspondence_distance_sq](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            const auto& [point, point_data] = points[i];
            const auto& neighbor = this->map_.closest(point, point_data, [this, &point_data](const VoxelMap<Wrapper>::Neighbor& neighbor) {
                return neighbor.data.prediction == point_data.prediction && neighbor.weight > this->correspondence_threshold_[point_data.prediction];
            }, correspondence_distance_sq);
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
                    return System(J_r.transpose() * target.weight * J_r, J_r.transpose() * target.weight * residual);
                });
        }, sum);

    return JTJ.ldlt().solve(-JTr);
}

std::pair<Sophus::SE3f, Sophus::SE3f> OdometrySystem::align(Wrapper& points, const Sophus::SE3f& delta, Profiler& profiler) {
    // determine starting point
    Sophus::SE3f initial_guess = last_pose_ * delta;
    if(map_.empty()) return std::make_pair(initial_guess, Sophus::SE3f{});
    points.transform(initial_guess);
    // compute adaptive threshold
    float sigma = threshold_.compute();
    // retrieve fields ahead of time to avoid repeated queries
    // perform ICP
    Sophus::SE3f T_icp{};
    PROFILE(align, &profiler) {
        for(size_t j = 0; j < config_.max_iterations; ++j) {
            // accumulate iteration count
            PROFILE_SKIP(align) statistics.iterations++;
            CorrespondenceSet correspondences; 
            PROFILE(iteration, &align) {
                // get correspondences
                correspondences = OdometrySystem::correspond(points, sigma);
                // update statistics
                PROFILE_SKIP(iteration) statistics.correspondences += static_cast<float>(correspondences.size());    
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
        "\tcorrespondence_target: " << config.correspondence.target << "," << std::endl <<
        "\tcorrespondence_threshold: " << format_threshold(config.correspondence.threshold);
}