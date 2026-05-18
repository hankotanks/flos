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
void deskew_frame_inner(util::CloudView& points, const std::string& name, const Sophus::SE3f &motion) noexcept {
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

void deskew_frame(util::CloudView& points, const Sophus::SE3f &motion) noexcept {
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

// TODO: Move this into /util
void transform(util::CloudView& cloud, const Sophus::SE3f& t) {
    for(size_t i = 0; i < cloud.size(); ++i) {
        cloud[i].point() = t * cloud[i].point();
    }
}
} // private scope

const Sophus::SE3f& OdometrySystem::register_to_map(util::CloudView&& points) noexcept {
    // deskewing
    deskew_frame(points, last_delta_);
    // sort the points by confidence (highest first)
    points.sort<float>("confidence", [](const float& fst, const float& snd) { return fst > snd; });
    // downsample incoming points to obtain keypoints and the preprocessed frame
    util::CloudView points_kp{points};
    util::voxel_downsample(points_kp, config_.map.voxel_size);
    // compute pose w/ ICP
    const auto [pose, deviation] = OdometrySystem::align(points_kp);
    threshold_.update(deviation);
    // append points to map
    // TODO: When appending to map, we need to prefer high confidence points
    map_.update(std::move(points), pose);
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

OdometrySystem::Semantics::Semantics(const util::CloudView& points) : 
    field_class(points.field<uint8_t>("class")),
    field_confidence(points.field<float>("confidence")) { /* STUB */ }

Eigen::Matrix<float, 6, 1> OdometrySystem::solve(const util::CloudView& points, double sigma, const OdometrySystem::Semantics& semantics) const noexcept {
    using Correspondences = tbb::concurrent_vector<std::pair<util::CloudView::ConstPoint, util::CloudView::ConstPoint>>;

    // this predicate only allows matching between points with the same semantic information
    VoxelMap::Predicate pred = [&semantics](const util::CloudView::ConstPoint& point, const util::CloudView::ConstPoint& query) -> bool {
        return semantics.field_class[point.index] != semantics.field_class[query.index];
    };

    Correspondences correspondences;
    correspondences.reserve(points.size());
    double correspondence_distance_max = sigma * config_.registration.correspondence_multiplier;
    tbb::parallel_for(tbb::blocked_range<size_t>{0, points.size()}, [&](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            const auto& maybe = map_.closest(points[i], pred);
            if(maybe) {
                const auto& [neighbor, distance] = *maybe;
                if(distance >= correspondence_distance_max) continue;
                correspondences.emplace_back(points[i], neighbor);
            }
        }
    });

    auto compute_jacobian_and_residual = [](const auto &correspondence) {
        const auto& [source, target] = correspondence;
        const Eigen::Vector3f residual = source.point() - target.point();
        Eigen::Matrix<float, 3, 6> J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3f::hat(source.point());
        return std::make_tuple(J_r, residual);
    };

    using System = std::pair<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>>;

    auto sum_linear_systems = [](System a, const System &b) {
        a.first += b.first;
        a.second += b.second;
        return a;
    };

    auto GM_weight = [&](const double &residual2) {
        return (sigma * sigma) / ((sigma + residual2) * (sigma + residual2));
    };

    using correspondence_iterator = Correspondences::const_iterator;
    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        tbb::blocked_range<correspondence_iterator>{correspondences.cbegin(), correspondences.cend()},
        System(Eigen::Matrix<float, 6, 6>::Zero(), Eigen::Matrix<float, 6, 1>::Zero()),
        [&](const tbb::blocked_range<correspondence_iterator>& range, System J) -> System {
            return std::transform_reduce(
                range.begin(), range.end(), J, sum_linear_systems, [&](const auto& correspondence) {
                    const auto& [J_r, residual] = compute_jacobian_and_residual(correspondence);
                    const util::CloudView::ConstPoint& point = correspondence.first;
                    // weight the kernel based on confidence from the segmentation model
                    const double w = semantics.field_confidence[point.index] * GM_weight(residual.squaredNorm());
                    return System(J_r.transpose() * w * J_r, J_r.transpose() * w * residual);
                });
        }, sum_linear_systems);

    return JTJ.ldlt().solve(-JTr);
}

std::pair<Sophus::SE3f, Sophus::SE3f> OdometrySystem::align(const util::CloudView& points) const {
    // determine starting point
    const auto initial_guess = last_pose_ * last_delta_;
    if(map_.empty()) return std::make_pair(initial_guess, Sophus::SE3f{});
    // compute adaptive threshold
    double sigma = threshold_.compute();
    // copy the frame for use in ICP
    util::CloudView source{points};
    transform(source, initial_guess);
    // retrieve fields ahead of time to avoid repeated queries
    Semantics source_semantics{source};
    // perform ICP
    Sophus::SE3f T_icp = Sophus::SE3f();
    for(size_t j = 0; j < config_.registration.max_iterations; ++j) {
        // Equation (11)
        const Eigen::Matrix<float, 6, 1> dx = OdometrySystem::solve(source, sigma, source_semantics);
        const Sophus::SE3f est = Sophus::SE3f::exp(dx);
        // Equation (12)
        transform(source, est);;
        // Update iterations
        T_icp = est * T_icp;
        // Termination criteria
        if (dx.norm() < config_.registration.convergence_criterion) break;
    }
    Sophus::SE3f pose = T_icp * initial_guess;
    // return transformation and deviation
    return std::make_pair(pose, initial_guess.inverse() * pose);
}
