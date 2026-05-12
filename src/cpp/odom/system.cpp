#include "system.h"

// system
#include <algorithm>
#include <optional>
// tbb
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>
// util
#include "util/cloud.h"

namespace {
static constexpr std::array<std::string, 4> names = {"timestamps", "time", "timestamp", "t"};
template<typename T>
std::optional<std::string> get_timestamp_field_name(util::Cloud& points) noexcept {
    for(const std::string& name : names) {
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
void deskew_frame_inner(util::Cloud& points, const std::string& name, const Sophus::SE3f &motion) noexcept {
    const std::vector<T>& timestamps = points.field<T>(name);
    const auto& [t_min, t_max] = std::minmax_element(timestamps.cbegin(), timestamps.cend());
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

void deskew_frame(util::Cloud& points, const Sophus::SE3f &motion) noexcept {
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

inline Eigen::Vector3i point_to_voxel(const Eigen::Vector3f& point, double voxel_size) {
    return {
        static_cast<int>(std::floor(point.x() / voxel_size)),
        static_cast<int>(std::floor(point.y() / voxel_size)),
        static_cast<int>(std::floor(point.z() / voxel_size))
    };
}

// TODO: I can avoid another cloud allocation if I implement Point::remove and a reverse Iterator for Cloud
void voxel_downsample(util::Cloud& cloud, const double voxel_size) {
    tsl::robin_map<Eigen::Vector3i, size_t> grid;
    grid.reserve(cloud.size());
    const std::vector<Eigen::Vector3f>& points = cloud.points();
    for(size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3i voxel = point_to_voxel(points[i], voxel_size);
        if(!grid.contains(voxel)) grid.insert({voxel, i});
    }
    util::Cloud cloud_original{std::move(cloud)};
    std::for_each(grid.cbegin(), grid.cend(), [&](const auto& voxel_and_index) {
        cloud.add_point(cloud_original, voxel_and_index.second);
    });
}

void transform(util::Cloud& cloud, const Sophus::SE3f& t) {
    for(size_t i = 0; i < cloud.size(); ++i) {
        cloud[i].point() = t * cloud[i].point();
    }
    // std::for_each(cloud.begin(), cloud.end(), [t](util::Cloud::Point point) {
    //     point.point() = t * point.point();
    // });
}
} // private scope

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

void OdometrySystem::register_to_map(util::Cloud&& points) noexcept {
    deskew_frame(points, last_delta_);

    voxel_downsample(points, config_.map.voxel_size * 0.5);

    util::Cloud points_source{points};
    voxel_downsample(points_source, config_.map.voxel_size * 1.5);

    const auto [pose, deviation] = OdometrySystem::align_to_map(points_source);

    threshold_.update(deviation);
    map_.update(std::move(points), pose);
    last_delta_ = last_pose_.inverse() * pose;
    last_pose_ = pose;
}

OdometrySystem::Correspondences OdometrySystem::correspondences(const util::Cloud& points, double sigma) const {
    Correspondences correspondences;
    correspondences.reserve(points.size());
    double correspondence_distance_max = sigma * config_.registration.correspondence_distance_multiplier;
    tbb::parallel_for(tbb::blocked_range<size_t>{0, points.size()}, [&](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            const auto& maybe = map_.closest(points[i]);
            if(maybe) {
                const auto& [neighbor, distance] = *maybe;
                if(distance < correspondence_distance_max) {
                    correspondences.emplace_back(points[i], neighbor);
                }
            }
            
        }
    });
    return correspondences;
}

OdometrySystem::LinearSystem OdometrySystem::build_linear_system(const OdometrySystem::Correspondences& correspondences, double sigma) const noexcept {
    auto compute_jacobian_and_residual = [](const auto &correspondence) {
        const auto& [source, target] = correspondence;
        const Eigen::Vector3f residual = source.point() - target.point();
        Eigen::Matrix<float, 3, 6> J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3f::hat(source.point());
        return std::make_tuple(J_r, residual);
    };

    auto sum_linear_systems = [](OdometrySystem::LinearSystem a, const OdometrySystem::LinearSystem &b) {
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
        LinearSystem(Eigen::Matrix<float, 6, 6>::Zero(), Eigen::Matrix<float, 6, 1>::Zero()),
        [&](const tbb::blocked_range<correspondence_iterator>& range, LinearSystem J) -> LinearSystem {
            return std::transform_reduce(
                range.begin(), range.end(), J, sum_linear_systems, [&](const auto& correspondence) {
                    const auto& [J_r, residual] = compute_jacobian_and_residual(correspondence);
                    const double w = GM_weight(residual.squaredNorm());
                    return LinearSystem(J_r.transpose() * w * J_r, J_r.transpose() * w * residual);
                });
        }, sum_linear_systems);

    return {JTJ, JTr};
}

std::pair<Sophus::SE3f, Sophus::SE3f> OdometrySystem::align_to_map(const util::Cloud& points) const {
    const auto initial_guess = last_pose_ * last_delta_;
    if(map_.empty()) return std::make_pair(initial_guess, Sophus::SE3f{});
    double sigma = threshold_.compute();
    util::Cloud source{points};
    transform(source, initial_guess);
    Sophus::SE3f T_icp = Sophus::SE3f();
    for(size_t j = 0; j < config_.registration.max_iterations; ++j) {
        // Equation (10)
        const auto correspondences = OdometrySystem::correspondences(source, sigma);
        // Equation (11)
        const auto &[JTJ, JTr] = OdometrySystem::build_linear_system(correspondences, sigma);
        const Eigen::Matrix<float, 6, 1> dx = JTJ.ldlt().solve(-JTr);
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
