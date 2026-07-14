#include "rerun_node.h"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RerunNode>();
    node->run();
    rclcpp::shutdown();
    return 0;
}
