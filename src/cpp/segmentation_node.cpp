// ros
#include <rclcpp/logging.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
// package
#include "segmentation_node_config.h"
#include "segmentation/system.h"
// util
#include "util/buffer_node.h"
#include "util/cloud.h"
#include "util/profiler.h"

// PARAMETERS
// debug
// incoming_points_topic
// path_to_model
// voxel_size
// max_distance
// ignore

// TOPICS
// sub: incoming_points_topic (default: $/raw/points) [PointCloud2]
// pub: $/inference [PointCloud2]

class SegmentationNode : 
    util::Profiler, 
    public BufferNode<std::pair<util::Cloud, std_msgs::msg::Header>, sensor_msgs::msg::PointCloud2> {
public:
    explicit SegmentationNode() : util::Profiler(NODE_NAME),
        BufferNode("incoming_points_topic", "/" PROJECT_NAME "/raw/points"),
        config_(this),
        system_(config_.config) {
        if(!rclcpp::ok()) return;
        // point publisher (subscriber is now handled by BufferNode)
        outgoing_points_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/" PROJECT_NAME "/inference", 10);
        // start the worker
        BufferNode::start();
    }
public:
    SegmentationNode(const SegmentationNode&) = delete;
    SegmentationNode(SegmentationNode&&) = delete;
    SegmentationNode& operator=(const SegmentationNode&) = delete;
    SegmentationNode& operator=(SegmentationNode&&) = delete;
    ~SegmentationNode() { 
        BufferNode::stop(); 
    }
protected:
    
protected:
    // config
    SegmentationNodeConfig config_;
    // system
    SegmentationSystem system_;
    // topic stuff  
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr outgoing_points_publisher_;
protected:
    virtual void loop() noexcept override final {
        py::scoped_interpreter guard{};
        // initialize the system
        SegmentationSystem::Model model = system_.load(config_.path_to_model);
        RCLCPP_INFO_ONCE(this->get_logger(), NODE_NAME " successfully loaded inference model");
        // loop until node is stopped
        while(rclcpp::ok()) {
            // get the next batch of points
            auto maybe = BufferNode::pop();
            if(!maybe) break;
            auto [cloud, header] = std::move(*maybe);
            static_assert(std::is_same_v<decltype(cloud), util::Cloud>);
            static_assert(std::is_same_v<decltype(header), std_msgs::msg::Header>);
            { // perform the inference
                util::Profiler profiler("inference", this);
                model.infer(cloud);
            }
            // publish the result
            outgoing_points_publisher_->publish(std::move(cloud.message(header)));
        }
    }

    // overload of the BufferNode::read method
    virtual std::pair<util::Cloud, std_msgs::msg::Header> read(const std::shared_ptr<sensor_msgs::msg::PointCloud2>& msg) noexcept final override {
        util::Cloud cloud{msg};
        system_.preprocess(cloud);
        return std::make_pair(std::move(cloud), msg->header);
    }
};

// TODO: this node specifically causes segfaults on SIGINT
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SegmentationNode>();
    rclcpp::spin(std::static_pointer_cast<rclcpp::Node>(node));
    rclcpp::shutdown();
    return 0;
}
