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
std::optional<T> optional_positive_parameter(rclcpp::Node* node, const std::string& name, T to) {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
    rcl_interfaces::msg::ParameterDescriptor desc{};
    desc.dynamic_typing = true;
    if constexpr (std::is_integral_v<T>) {
        rcl_interfaces::msg::IntegerRange range{};
        range.from_value = 0;
        range.to_value = to;
        desc.integer_range.push_back(range);
    } else if constexpr (std::is_floating_point_v<T>) {
        rcl_interfaces::msg::FloatingPointRange range{};
        range.from_value = 0.0;
        range.to_value = to;
        desc.floating_point_range.push_back(range);
    }
    node->declare_parameter(name, rclcpp::ParameterValue{}, desc);
    T value;
    if(node->get_parameter(name, value)) return value;
    return std::nullopt;
}

// TODO: Consider the upper bounds of these parameters
OdometrySystem::Config configure(rclcpp::Node* node) {
    OdometrySystem::Config config;
    // map
    auto map_max_distance = optional_positive_parameter(node, "max_distance", std::numeric_limits<double>::max());
    if(map_max_distance) config.map.max_distance = *map_max_distance;
    auto map_voxel_size = optional_positive_parameter(node, "voxel_size", std::numeric_limits<double>::max());
    if(map_voxel_size) config.map.voxel_size = *map_voxel_size;
    auto map_voxel_capacity = optional_positive_parameter(node, "voxel_capacity", std::numeric_limits<size_t>::max());
    if(map_voxel_capacity) config.map.voxel_capacity = *map_voxel_capacity;
    // threshold
    auto threshold_initial = optional_positive_parameter(node, "initial", std::numeric_limits<double>::max());
    if(threshold_initial) config.threshold.initial = *threshold_initial;
    auto threshold_min_motion = optional_positive_parameter(node, "min_motion", std::numeric_limits<double>::max());
    if(threshold_min_motion) config.threshold.min_motion = *threshold_min_motion;
    // registration
    auto registration_max_iterations = optional_positive_parameter(node, "max_iterations", std::numeric_limits<size_t>::max());
    if(registration_max_iterations) config.registration.max_iterations = *registration_max_iterations;
    auto registration_max_threads = optional_positive_parameter(node, "max_threads", std::numeric_limits<size_t>::max());
    if(registration_max_threads) config.registration.max_threads = *registration_max_threads;
    auto registration_convergence_criterion = optional_positive_parameter(node, "convergence_criterion", std::numeric_limits<double>::max());
    if(registration_convergence_criterion) config.registration.convergence_criterion = *registration_convergence_criterion;
    auto registration_correspondence_distance_multiplier = optional_positive_parameter(node, "correspondence_distance_multiplier", std::numeric_limits<size_t>::max());
    if(registration_correspondence_distance_multiplier) config.registration.correspondence_distance_multiplier = *registration_correspondence_distance_multiplier;
    return config;
}

} // private scope

class OdometryNode : public BufferNode<util::Cloud, sensor_msgs::msg::PointCloud2> {
public:
    explicit OdometryNode() : BufferNode("incoming_points_topic", "/" PROJECT_NAME "/inference"), system(configure(this)) {
        this->declare_parameter<std::string>("local_map_topic", "/" PROJECT_NAME "/local_map");
        this->get_parameter("local_map_topic", local_map_topic);
        local_map_publisher = this->create_publisher<sensor_msgs::msg::PointCloud2>(local_map_topic, 10);  
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
    std::string local_map_topic;
    // publisher for local map
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_publisher;
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
            auto stamp = std::chrono::system_clock::now().time_since_epoch();
            std_msgs::msg::Header header;
            header.frame_id = "map";
            header.stamp.sec = std::chrono::duration_cast<std::chrono::seconds>(stamp).count();
            header.stamp.nanosec = std::chrono::duration_cast<std::chrono::nanoseconds>(stamp).count() % 1000000000;;
            local_map_publisher->publish(std::move(system.cloud().message(header)));
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