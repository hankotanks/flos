// ros
#include <rclcpp/rclcpp.hpp>
// package
#include "node.h"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SegmentationNode>();
    rclcpp::spin(std::static_pointer_cast<rclcpp::Node>(node));
    rclcpp::shutdown();
    return 0;
}