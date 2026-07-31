// ros
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
// package
#include "odometry/system.h"
// util
#include "util/rosarg.h"

struct OdometryNodeConfig {
    OdometrySystem::Config config;
    OdometryNodeConfig(rclcpp::Node* node) {
        util::param_opt(node, "max_distance", config.max_distance);
        util::param_opt(node, "voxel_size", config.voxel_size);
        util::param_opt(node, "voxel_capacity", config.voxel_capacity);
        // threshold
        util::param_opt(node, "threshold_initial", config.threshold.initial);
        util::param_opt(node, "threshold_motion", config.threshold.motion);
        // registration
        util::param_opt(node, "deskew", config.deskew);
        util::param_opt(node, "fixed_dt", config.fixed_dt);
        util::param_opt(node, "max_iterations", config.max_iterations);
        util::param_opt(node, "max_threads", config.max_threads);
        util::param_opt(node, "convergence_criterion", config.convergence_criterion);
        // correspondence
        util::param_opt(node, "correspondence_weighting", config.correspondence.weighting);
        util::param_opt(node, "correspondence_weighting_by_belief", config.correspondence.weighting_by_belief);
        util::param_opt(node, "correspondence_target", config.correspondence.target);
        util::param_opt(node, "correspondence_threshold", config.correspondence.threshold);
        // log parameters
        RCLCPP_INFO_STREAM(node->get_logger(), config);
    }
};