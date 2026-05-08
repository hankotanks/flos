#include "buffer_node.h"

// ros
#include <rcl_interfaces/msg/parameter_descriptor.hpp>

BufferNode::BufferNode() : rclcpp::Node(PROJECT_NAME), voxel_size(std::nullopt) {
    // declare parameters
    rcl_interfaces::msg::ParameterDescriptor voxel_size_desc{};
    voxel_size_desc.dynamic_typing = true;
    this->declare_parameter("voxel_size", rclcpp::ParameterValue{}, voxel_size_desc);
    this->declare_parameter<std::string>("incoming_points_topic", "/hesai/pandar");
    // get parameters
    double voxel_size_temp; // NOTE: extra careful because I'm not sure of get_parameter's type inferencing behavior
    if(this->get_parameter("voxel_size", voxel_size_temp)) voxel_size = voxel_size_temp;
    this->get_parameter("incoming_points_topic", incoming_points_topic);
    // create subscriber
    using PointCloud2 = sensor_msgs::msg::PointCloud2;
    incoming_points_subscriber_ = this->create_subscription<PointCloud2>(incoming_points_topic, 10,
        [this](const PointCloud2::SharedPtr msg) { this->push(msg); });
}

BufferNode::~BufferNode() {
    { // stop
        std::lock_guard lock(mutex_);
        running_ = false; 
    }
    cv_.notify_all(); 
}

void BufferNode::push(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) noexcept {
    // TODO: voxel downsampling w.r.t. voxel_size
    if(msg->width * msg->height == 0) return;
    { // ensure capacity isn't reached
        std::unique_lock lock(mutex_);
        if(buffer_.size() >= capacity) {
            RCLCPP_WARN_ONCE(this->get_logger(), "segmentation_node can't keep up!");
            return;
        }
        // push the cloud
        buffer_.emplace(msg, msg->header);
        cv_.notify_one();
    }   
}

std::optional<BufferNode::Entry> BufferNode::pop() noexcept {
    { // wait
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return !running_ || !buffer_.empty(); });
        if(!running_ && buffer_.empty()) return std::nullopt; // exhausted
    }
    // pop points from buffer and return
    auto cloud_and_header = std::move(buffer_.front());
    buffer_.pop();
    return cloud_and_header;
}

void BufferNode::wait_until_drained() noexcept {
    while(rclcpp::ok()) {
        { // wait for buffer to drain
            std::lock_guard<std::mutex> lock(mutex_);
            if(buffer_.empty() && !active.load()) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
