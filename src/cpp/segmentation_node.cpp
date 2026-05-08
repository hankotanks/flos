// ros
#include <rclcpp/rclcpp.hpp>
// forest_segmentation
#include "base_node.h"

class SegmentationNode : public BaseNode {
public:
    SegmentationNode() : BaseNode() { /* STUB */ }
    SegmentationNode(const SegmentationNode&) = delete;
    SegmentationNode(SegmentationNode&&) = delete;
    SegmentationNode& operator=(const SegmentationNode&) = delete;
    SegmentationNode& operator=(SegmentationNode&&) = delete;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SegmentationNode>();
    rclcpp::spin(std::static_pointer_cast<rclcpp::Node>(node));
    rclcpp::shutdown();
    return 0;
}
