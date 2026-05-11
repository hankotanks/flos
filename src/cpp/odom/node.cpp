// system
#include <limits>
// ros
#include <rclcpp/rclcpp.hpp>
// package
#include "system.h"
// util
#include "util/buffer_node.h"
#include "util/cloud.h"

namespace {
template<typename T>
std::optional<T> optional_positive_parameter(rclcpp::Node* node, const std::string& name) {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
    rcl_interfaces::msg::ParameterDescriptor desc{};
    desc.dynamic_typing = true;
    if constexpr (std::is_integral_v<T>) {
        rcl_interfaces::msg::IntegerRange range{};
        range.from_value = 0;
        range.to_value = std::numeric_limits<T>::max();
        desc.integer_range.push_back(range);
    } else if constexpr (std::is_floating_point_v<T>) {
        rcl_interfaces::msg::FloatingPointRange range{};
        range.from_value = 0.0;
        range.to_value = std::numeric_limits<T>::max();
        desc.floating_point_range.push_back(range);
    }
    node->declare_parameter(name, rclcpp::ParameterValue{}, desc);
    T value;
    if(node->get_parameter(name, value)) return value;
    return std::nullopt;
}
} // private scope

class OdometryNode : public BufferNode<util::Cloud, sensor_msgs::msg::PointCloud2> {
public:
    explicit OdometryNode() : BufferNode("incoming_points_topic", "/" PROJECT_NAME "/inference"), system(
        optional_positive_parameter<double>(this, "max_distance"),
        optional_positive_parameter<double>(this, "voxel_size"),
        optional_positive_parameter<size_t>(this, "voxel_capacity")) {        
        odometry_thread = std::thread(&OdometryNode::loop, this);
    }
public:
    OdometryNode(const OdometryNode&) = delete;
    OdometryNode(OdometryNode&&) = delete;
    OdometryNode& operator=(const OdometryNode&) = delete;
    OdometryNode& operator=(OdometryNode&&) = delete;
    ~OdometryNode() {
        if(odometry_thread.joinable()) odometry_thread.join();
    }

protected:
    // core of the node
    OdometrySystem system;
    // separate thread for processing
    std::thread odometry_thread;
protected:
    void loop() {
        while(rclcpp::ok()) {
            // get the next batch of points
            auto maybe = BufferNode::pop();
            if(!maybe) break;
            active = true;
            system.register_to_map(std::move(*maybe));
            active = false;
        }
    }
    
    virtual util::Cloud read(const std::shared_ptr<sensor_msgs::msg::PointCloud2>& msg) const noexcept final override { return msg; }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OdometryNode>();
    rclcpp::spin(std::static_pointer_cast<rclcpp::Node>(node));
    rclcpp::shutdown();
    return 0;
}