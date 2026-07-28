// ros
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
// package
#include "odometry/system.h"
// util
#include "util/rosargs.h"

struct OdometryNodeConfig {
    OdometrySystem::Config config;
    OdometryNodeConfig(rclcpp::Node* node) {
        util::param_opt(node, "max_distance", config.map.max_distance);
        util::param_opt_map(node, "min_weight", config.map.min_weight);
        util::param_opt(node, "voxel_size", config.map.voxel_size);
        util::param_opt(node, "voxel_capacity", config.map.voxel_capacity);
        // threshold
        util::param_opt(node, "threshold_initial", config.threshold.threshold_initial);
        util::param_opt(node, "min_motion", config.threshold.min_motion);
        // registration
        util::param_opt(node, "deskew", config.registration.deskew);
        util::param_opt(node, "confidence_weighting", config.registration.confidence_weighting);
        util::param_opt(node, "correspondence_restriction", config.registration.correspondence_restriction);
        util::param_opt(node, "correspondence_with_voxel_belief", config.registration.correspondence_with_voxel_belief);
        util::param_opt(node, "constant_velocity", config.registration.constant_velocity);
        util::param_opt(node, "max_iterations", config.registration.max_iterations);
        util::param_opt(node, "max_threads", config.registration.max_threads);
        util::param_opt(node, "convergence_criterion", config.registration.convergence_criterion);
        util::param_opt(node, "correspondence_distance_multiplier", config.registration.correspondence_multiplier);
        // log parameters
        RCLCPP_INFO_STREAM(node->get_logger(), config);
    }
};