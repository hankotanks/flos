#include "buffer_node.h"

// ros
#include <tsl/robin_map.h>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>

template <>
struct std::hash<Eigen::Vector3i> {
    std::size_t operator()(const Eigen::Vector3i& voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
    }
};

namespace {
inline Eigen::Vector3i point_to_voxel(const Eigen::Vector3f& point, double voxel_size) {
    return {
        static_cast<int>(std::floor(point.x() / voxel_size)),
        static_cast<int>(std::floor(point.y() / voxel_size)),
        static_cast<int>(std::floor(point.z() / voxel_size))
    };
}

void voxel_downsample(util::Cloud &cloud, const double voxel_size) {
    tsl::robin_map<Eigen::Vector3i, size_t> grid;
    grid.reserve(cloud.size());
    const std::vector<Eigen::Vector3f>& points = cloud.points();
    for(size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3i voxel = point_to_voxel(points[i], voxel_size);
        if(!grid.contains(voxel)) grid.insert({voxel, i});
    }
    util::Cloud cloud_original{std::move(cloud)};
    std::for_each(grid.cbegin(), grid.cend(), [&](const auto& voxel_and_index) {
        cloud.add_point(cloud_original, voxel_and_index.second);
    });
}
} // private scope

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
    if(msg->width * msg->height == 0) return;
    { // ensure capacity isn't reached
        std::unique_lock lock(mutex_);
        if(buffer_.size() >= capacity) {
            RCLCPP_WARN_ONCE(this->get_logger(), "segmentation_node can't keep up!");
            return;
        }
        // push the cloud
        buffer_.emplace(msg, msg->header);
        // downsample if voxel_size is specified
        if(voxel_size) {
            util::Cloud& cloud = buffer_.back().first;
            voxel_downsample(cloud, *voxel_size);
        }
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
