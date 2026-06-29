#include "system.h"

// system
#include <algorithm>
#include <optional>
#include <span>
// tbb
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>
// util
#include "util/cloud.h"

namespace {
// field names to search in get_timestamp_field_name
static constexpr std::array<std::string, 4> timestamp_field_names = {"timestamps", "time", "timestamp", "t"};

template<typename T>
std::optional<std::string> get_timestamp_field_name(util::CloudView& points) noexcept {
    for(const std::string& name : timestamp_field_names) {
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
void deskew_frame_inner(util::CloudView& points, const std::string& name, const Sophus::SE3f& motion) noexcept {
    std::span<const T> timestamps = points.field<T>(name);
    const auto& [t_min, t_max] = std::minmax_element(timestamps.begin(), timestamps.end());
    const auto t_norm = [&](const double t) { 
        return (t - *t_min) / (*t_max - *t_min); 
    };
    const auto& omega = motion.log();
    tbb::parallel_for(tbb::blocked_range<size_t>{0, points.size()}, [&](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            const T stamp = t_norm(points[i].field<T>(name));
            const auto pose = Sophus::SE3f::exp((static_cast<float>(stamp) - 1.f) * omega);
            points[i].point() = pose * points[i].point();
        };
    });
}

void deskew_frame(util::CloudView& points, const Sophus::SE3f& motion) noexcept {
    auto field_name = get_timestamp_field_name<float>(points);
    if(field_name) {
        deskew_frame_inner<float>(points, *field_name, motion);
        return;
    }
    field_name = get_timestamp_field_name<double>(points);
    if(field_name) {
        deskew_frame_inner<double>(points, *field_name, motion);
    }
}
} // private scope

OdometrySystem::OdometrySystem(const Config& config) : config_(config),
    map_(config.map.max_distance, config.map.voxel_size, config.map.voxel_capacity),
    threshold_(config.map.max_distance, config.threshold.threshold_initial, config.threshold.min_motion) {
    size_t max_threads = config.registration.max_threads > 0 ? \
        config.registration.max_threads : tbb::this_task_arena::max_concurrency();
    volatile static const tbb::global_control _ = tbb::global_control( // NOTE: static storage type is required here
        tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_threads));
}

const Sophus::SE3f& OdometrySystem::register_to_map(util::CloudView& points) noexcept {
    // deskewing
    deskew_frame(points, last_delta_);
    // downsample incoming points to obtain keypoints and the preprocessed frame
    util::CloudView points_kp{points};
    util::voxel_downsample(points_kp, config_.map.voxel_size * OdometrySystem::VOXEL_SIZE_SCALAR);
    Wrapper points_kp_wrapper{points_kp};
    // compute pose w/ ICP
    const auto [pose, deviation] = OdometrySystem::align(points_kp_wrapper);
    threshold_.update(deviation);
    // append points to map
    Wrapper points_wrapper{points}; // TODO: Semantics is constructed twice, once in OdometrySystem::align and once in this
    map_.update(points_wrapper, pose);
    // store new pose
    last_delta_ = last_pose_.inverse() * pose;
    last_pose_ = pose;
    // return reference to the current pose
    return last_pose_;
}

void OdometrySystem::AdaptiveThreshold::update(const Sophus::SE3f& deviation) noexcept {
    const double model_error = [&]() {
        const double theta = Eigen::AngleAxisf(deviation.rotationMatrix()).angle();
        const double delta_rot = 2.0 * max_distance_ * std::sin(theta / 2.0);
        const double delta_trans = deviation.translation().norm();
        return delta_trans + delta_rot;
    }();
    if (model_error > min_motion_) {
        model_sse_ += model_error * model_error;
        num_sample_++;
    }
}

Eigen::Matrix<float, 6, 1> OdometrySystem::solve(const Wrapper& points, double sigma) const noexcept {
    using Correspondence = std::tuple<Eigen::Vector3f, Semantics, VoxelMap<Wrapper>::Neighbor>;

    // this predicate only allows matching between points with the same semantic information
    static const VoxelMap<Wrapper>::Predicate pred = [this]() -> VoxelMap<Wrapper>::Predicate {
        if(this->config_.registration.correspondence_restriction) {
            if(this->config_.registration.correspondence_with_voxel_belief) {
                return [](const Belief& data, Semantics, Semantics query) -> bool {
                    return query.prediction == data.prediction;
                };
            } else {
                return [](const Belief&, Semantics point, Semantics query) -> bool { 
                    return query.prediction == point.prediction;
                };
            }
        } else {
            return [](const Belief&, Semantics, Semantics) -> bool { return true; };
        }
    }();
    
    tbb::concurrent_vector<Correspondence> correspondences;
    correspondences.reserve(points.size());
    double correspondence_distance_max = sigma * config_.registration.correspondence_multiplier;
    tbb::parallel_for(tbb::blocked_range<size_t>{0, points.size()}, [&](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            const auto& [point, point_data] = points[i];
            const auto& maybe = map_.closest(point, point_data, pred);
            if(maybe) {
                const auto& neighbor = *maybe;
                if(neighbor.distance >= correspondence_distance_max) continue;
                correspondences.emplace_back(point, point_data, neighbor);
            }
        }
    });

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
                    float w = config_.registration.confidence_weighting ? target.weight : 1.0;
                    return System(J_r.transpose() * w * J_r, J_r.transpose() * w * residual);
                });
        }, system_sum);

    return JTJ.ldlt().solve(-JTr);
}

std::pair<Sophus::SE3f, Sophus::SE3f> OdometrySystem::align(Wrapper& points) const {
    // determine starting point
    const auto initial_guess = last_pose_ * last_delta_;
    if(map_.empty()) return std::make_pair(initial_guess, Sophus::SE3f{});
    points.transform(initial_guess);
    // compute adaptive threshold
    double sigma = threshold_.compute();
    // retrieve fields ahead of time to avoid repeated queries
    // perform ICP
    Sophus::SE3f T_icp{};
    for(size_t j = 0; j < config_.registration.max_iterations; ++j) {
        // Equation (11)
        const Eigen::Matrix<float, 6, 1> dx = OdometrySystem::solve(points, sigma);
        const Sophus::SE3f est = Sophus::SE3f::exp(dx);
        // Equation (12)
        points.transform(est);
        // Update iterations
        T_icp = est * T_icp;
        // Termination criteria
        if(dx.norm() < config_.registration.convergence_criterion) break;
    }
    Sophus::SE3f pose = T_icp * initial_guess;
    // return transformation and deviation
    return std::make_pair(pose, initial_guess.inverse() * pose);
}
