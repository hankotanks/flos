#include "odometry/system.h"

// system
#include <algorithm>
#include <optional>
// tbb
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
// package
#include "odometry/adaptive_threshold.h"
#include "odometry/projection.h"
#include "odometry/time_system.h"
#include "classes.h"
// util
#include "util/cloud.h"
#include "util/profiler.h"

OdometrySystem::OdometrySystem(const Config& config) : util::Profiler("OdometrySystem"),
    config_(config),
    map_(config_.max_distance, config_.voxel_size, config_.voxel_capacity),
    projection_(config_.max_distance, config_.max_distance_ground, config_.voxel_size, config_.projection.iou, 
        std::sqrt(config_.voxel_size * config_.voxel_size / static_cast<float>(config_.voxel_capacity)),
        [this](const Semantics& semantics) { return this->config_.projection.ground.count(semantics.prediction) > 0; },
        [this](const Semantics& semantics) { return this->config_.projection.target.count(semantics.prediction) > 0; }),
    threshold_(config_.max_distance, config_.threshold.initial, config_.threshold.motion) {
    size_t max_threads = config_.max_threads > 0 ? \
        config_.max_threads : tbb::this_task_arena::max_concurrency();
    volatile static const tbb::global_control _ = tbb::global_control( // NOTE: static storage type is required here
        tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_threads)); 
}

const Sophus::SE3f& OdometrySystem::register_to_map(util::Cloud& points, TimeSystem::Stamp stamp) noexcept {
    if(config_.constant_scan_rate) stamp = TimeSystem::Stamp{};
    // reset statistics
    statistics_ = Statistics{};
    // perform registration
    PROFILE(registration, this) {
        // compute delta
        const auto prior = time_system_.predict_motion(points, last_delta_, stamp);
        // deskewing
        if(config_.deskew) {
            PROFILE(deskewing, &registration) {
                time_system_.deskew(points, prior);
            }
        }
        // downsample incoming points to obtain keypoints and the preprocessed frame
        util::Cloud points_kp{points};
        if(config_.double_downsample) {
            PROFILE(downsample, &registration) {
                points_kp.downsample(config_.voxel_size * OdometrySystem::VOXEL_SIZE_SCALAR);
            }
        }
        // get keypoints
        SemanticCloud points_kp_wrapper{points_kp};
        // prepare rough initial guess
        Sophus::SE3f initial_guess = last_pose_ * prior.frame_delta;
        points_kp_wrapper.transform(initial_guess);
        // initial_guess refinement
        PROFILE(align, &registration) {
            float sigma = threshold_.compute();
            float correspondence_distance = sigma * OdometrySystem::CORRESPONDENCE_DISTANCE_SCALAR;
            // refine initial_guess (invariant: point_kp_wrapper is transformed by initial_guess)
            if(config_.projection.enable) {
                PROFILE(refinement, &align) {
                    if(frame_ > config_.voxel_capacity) OdometrySystem::refine(points_kp_wrapper, initial_guess, correspondence_distance);
                }
            }
            // ICP
            Sophus::SE3f T_icp = OdometrySystem::align(points_kp_wrapper, initial_guess, correspondence_distance);
            // build the current estimated pose
            Sophus::SE3f pose = T_icp * initial_guess;
            threshold_.update(initial_guess.inverse() * pose);
            // update last pose
            last_delta_ = last_pose_.inverse() * pose;
            last_pose_ = pose;
            // maintain ground plane and previous frame's projection
            if(config_.projection.enable) {
                PROFILE(refinement_update, &align) {
                    if(frame_ > config_.voxel_capacity) {
                        projection_.fit(map_, last_pose_, T_icp);
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
            statistics_.velocity.linear = xi.tail<3>().norm() / static_cast<float>(prior.dt);
            statistics_.velocity.angular = xi.head<3>().norm() / static_cast<float>(prior.dt);
        }
    }
    ++frame_;
    // return const ref to the computed pose, since we store it anyway
    return last_pose_;
}

OdometrySystem::CorrespondenceSet OdometrySystem::correspond(const SemanticCloud& points, float correspondence_distance) const noexcept {
    tbb::concurrent_vector<Correspondence> correspondences;
    correspondences.reserve(points.size());
    // collect all correspondences
    float correspondence_distance_sq = correspondence_distance * correspondence_distance;
    tbb::parallel_for(tbb::blocked_range<size_t>{0, points.size()}, [this, &points, correspondence_distance_sq, &correspondences](const tbb::blocked_range<size_t>& range) {
        for(size_t i = range.begin(); i < range.end(); ++i) {
            const auto& [point, data] = points[i];
            const auto match = this->map_.adjacency(point, data, correspondence_distance_sq);
            if(match) correspondences.emplace_back(point, *match);
        }
    });
    // prune all low-weight correspondences down to a threshold...
    if(config_.correspondence.target > 0 && config_.correspondence.target < correspondences.size()) {
        std::nth_element(correspondences.begin(), correspondences.begin() + config_.correspondence.target, correspondences.end(),
            [](const Correspondence& fst, const Correspondence& snd) {
                const auto& [fst_point, fst_match] = fst;
                const auto& [snd_point, snd_match] = snd;
                return fst_match.weight > snd_match.weight;
            });
        correspondences.resize(config_.correspondence.target);
    }
    return correspondences;
}

std::pair<Eigen::Matrix<float, 6, 1>, float> OdometrySystem::solve(const OdometrySystem::CorrespondenceSet& correspondences) noexcept {
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
                    const auto& [source, match] = correspondence;
                    Eigen::Matrix<float, 3, 6> J_r;
                    J_r.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
                    J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3f::hat(source);
                    // weight the kernel based on confidence from the segmentation model
                    return System(J_r.transpose() * match.weight * J_r, J_r.transpose() * match.weight * match.residual, match.weight * match.distance_sq, match.weight);
                });
        }, sum);

    return std::make_pair(JTJ.ldlt().solve(-JTr), w > 0.f ? chi / w : 0.f);
}

Sophus::SE3f OdometrySystem::refine(SemanticCloud& points, Sophus::SE3f& initial_guess, float correspondence_distance) {
    if(auto result = projection_.align(points, config_.max_iterations, config_.convergence_criterion)) {
        auto T_bev = *result;
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
            statistics_.align.refined = true;
            return T_bev;
        }
    }
    return {};
}

Sophus::SE3f OdometrySystem::align(SemanticCloud& points, const Sophus::SE3f& initial_guess, float correspondence_distance) {
    // determine starting point
    if(map_.empty()) return {};
    // termination threshold (Ceres-style)
    float chi = -1.f;
    // perform ICP
    Sophus::SE3f T_icp{};
    for(size_t i = 0; i < config_.max_iterations; ++i) {
        // accumulate iteration count
        statistics_.align.iterations++;
        CorrespondenceSet correspondences; 
        // get correspondences
        correspondences = OdometrySystem::correspond(points, correspondence_distance);
        // update statistics
        statistics_.align.mean_correspondence_count += static_cast<float>(correspondences.size());    
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
    statistics_.align.mean_correspondence_count /= static_cast<float>(statistics_.align.iterations);
    statistics_.align.chi = chi;
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
        "\tdouble_downsample: " << config.double_downsample << "," << std::endl <<
        "\tconstant_scan_rate: " << config.constant_scan_rate << "," << std::endl <<
        "\tmax_iterations: " << config.max_iterations << "," << std::endl <<
        "\tmax_threads: " << config.max_threads << "," << std::endl <<
        "\tconvergence_criterion: " << config.convergence_criterion << "," << std::endl <<
        "\tcorrespondence_weighting: " << config.correspondence.weighting << "," << std::endl <<
        "\tcorrespondence_target: " << config.correspondence.target << "," << std::endl <<
        "\tprojection_enable: " << config.projection.enable << "," << std::endl <<
        "\tprojection_iou: " << config.projection.iou << "," << std::endl <<
        "\tprojection_ground: " << format(config.projection.ground) << "," << std::endl <<
        "\tprojection_target: " << format(config.projection.target);
}