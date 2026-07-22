// system
#include <filesystem>
// ros
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
// package
#include "segmentation/system.h"
// util
#include "util/rosargs.h"

struct SegmentationNodeConfig {
    SegmentationSystem::Config config;
    std::filesystem::path path_to_model;
    SegmentationNodeConfig(rclcpp::Node* node) {
        util::param_opt(node, "voxel_size", config.voxel_size);
        util::param_opt(node, "max_distance", config.max_distance);
        // DONE: I would like my system to be class-name agnostic, so I've moved these parameters here
        // however, the actual class numbers should not be hard-coded...
        // this is a problem for a different day
        util::param_opt_set(node, "ignore", config.ignore);
        // path to the inference model
        path_to_model = PROJECT_ROOT;
        while(path_to_model.filename() != PROJECT_NAME) 
            path_to_model = path_to_model.parent_path();
        path_to_model /= "models";
        path_to_model /= "default.ckpt";
        util::param_opt_path(node, "path_to_model", path_to_model, true, true);
        // log parameters
        RCLCPP_INFO_STREAM(node->get_logger(), config);
    }
};