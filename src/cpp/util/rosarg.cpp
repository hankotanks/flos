#include "util/rosarg.h"

namespace util {
bool param_opt_path(rclcpp::Node* node, const std::string& name, std::filesystem::path& value, bool exists, std::optional<bool> file, bool create_directories) {
    std::string path_raw;
    if(param_opt<std::string>(node, name, path_raw)) {
        std::filesystem::path path{path_raw};
        if(!std::filesystem::exists(path)) {
            if(exists) {
                RCLCPP_ERROR_STREAM_ONCE(node->get_logger(), name << " did not exist: " << path_raw);
                rclcpp::shutdown();
            }
            if(create_directories) {
                if(file) {
                    std::filesystem::create_directories((*file) ? path.parent_path() : path);
                } else throw std::logic_error("unable to create directories without knowing if the target is a file or directory!");
                
            }
        } else if(std::filesystem::is_directory(path) && file && (*file)) {
            RCLCPP_ERROR_STREAM_ONCE(node->get_logger(), name << " expected a file, but found a directory: " << path_raw);
            rclcpp::shutdown();
        }
        value = path;
        return true;
    }
    return false;
}

void param_req_path(rclcpp::Node* node, const std::string& name, std::filesystem::path& value, bool exists, std::optional<bool> file, bool create_directories) {
    if(!param_opt_path(node, name, value, exists, file, create_directories)) {
        RCLCPP_ERROR_STREAM_ONCE(node->get_logger(), "missing a required argument: " << name);
        rclcpp::shutdown();
    }
}
}