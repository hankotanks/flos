// system
#include <limits>
// ros
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <nav_msgs/msg/odometry.hpp>
// package
#include "system.h"
// util
#include "util/buffer_node.h"
#include "util/cloud.h"
#include "util/profiler.h"

class OdometryNode : util::Profiler, public BufferNode<std::pair<util::CloudView, std_msgs::msg::Header>, sensor_msgs::msg::PointCloud2> {
public:
    explicit OdometryNode() : util::Profiler(NODE_NAME),
        BufferNode("incoming_points_topic", "/" PROJECT_NAME "/inference"), 
        system(OdometryNode::configure(this)) {
        // declare parameters
        this->declare_parameter<std::string>("local_map_topic", "/" PROJECT_NAME "/local_map");
        this->declare_parameter<std::string>("local_map_frame", "odom_frame");
        this->declare_parameter<bool>("local_map", false);
        // store parameters
        this->get_parameter("local_map_topic", local_map_topic);
        this->get_parameter("local_map_frame", local_map_frame);
        this->get_parameter("local_map", local_map);
        // initialize map publisher
        local_map_publisher = this->create_publisher<sensor_msgs::msg::PointCloud2>(local_map_topic, 10);  
        // initialize odometry publisher
        odom_publisher = this->create_publisher<nav_msgs::msg::Odometry>("/" PROJECT_NAME "/odometry", 10);
        // broadcaster for poses
        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        // spin up thread for odometry
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
    // parameters (excluding those related to the system config itself)
    std::string local_map_topic;
    std::string local_map_frame;
    bool local_map;
protected:
    static constexpr double local_map_rate = 5.0; // number of seconds between map publications
    // core
    OdometrySystem system;
    // publisher for local map
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_publisher;
    // odometry publisher
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher;
    // pose broadcaster
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    // separate thread for processing
    std::thread odometry_thread;
protected:
    void loop() {
        rclcpp::Time local_map_last_publication = this->now();
        while(rclcpp::ok()) {
            // get the next batch of points
            auto maybe = BufferNode::pop();
            if(!maybe) break;
            active = true;
            auto [cloud, header] = std::move(*maybe);
            // perform registration
            Sophus::SE3f pose;
            {   
                util::Profiler* this_profiler = this;
                util::Profiler profiler("registration", this_profiler);
                pose = system.register_to_map(std::move(cloud));
            }
            { // send the transformation from the lidar frame to the local map frame
                geometry_msgs::msg::TransformStamped msg;
                msg.header.stamp = header.stamp;
                msg.header.frame_id = local_map_frame;
                msg.child_frame_id = header.frame_id;
                msg.transform.translation.x = pose.translation().x();
                msg.transform.translation.y = pose.translation().y();
                msg.transform.translation.z = pose.translation().z();
                Eigen::Quaternionf q(pose.so3().unit_quaternion());
                msg.transform.rotation.x = q.x();
                msg.transform.rotation.y = q.y();
                msg.transform.rotation.z = q.z();
                msg.transform.rotation.w = q.w();
                // publish the transform
                tf_broadcaster->sendTransform(std::move(msg));
            }
            { // odometry
                nav_msgs::msg::Odometry msg;
                msg.header.stamp = header.stamp;
                msg.header.frame_id = local_map_frame;
                msg.child_frame_id = header.frame_id;
                msg.pose.pose.position.x = pose.translation().x();
                msg.pose.pose.position.y = pose.translation().y();
                msg.pose.pose.position.z = pose.translation().z();
                Eigen::Quaterniond q(pose.so3().unit_quaternion());
                msg.pose.pose.orientation.x = q.x();
                msg.pose.pose.orientation.y = q.y();
                msg.pose.pose.orientation.z = q.z();
                msg.pose.pose.orientation.w = q.w();
                msg.pose.covariance.fill(0.0);
                msg.pose.covariance[0] = 0.1;
                msg.pose.covariance[7] = 0.1;
                msg.pose.covariance[14] = 0.1;
                msg.pose.covariance[21] = 0.1;
                msg.pose.covariance[28] = 0.1;
                msg.pose.covariance[35] = 0.1;
                odom_publisher->publish(std::move(msg));
            }
            // send the local map
            if(local_map) {
                if((this->now() - local_map_last_publication).seconds() >= OdometryNode::local_map_rate) {
                    header.frame_id = local_map_frame;
                    const util::CloudView& cloud = system.cloud();
                    local_map_publisher->publish(std::move(cloud.message(header)));
                    local_map_last_publication = this->now();
                }
            }
            active = false;
        }
    }
    
    virtual std::pair<util::CloudView, std_msgs::msg::Header> read(const std::shared_ptr<sensor_msgs::msg::PointCloud2>& msg) noexcept final override {
        util::CloudView cloud{msg};
        return std::make_pair(std::move(cloud), msg->header);
    }
private:
    // helper function for range bounded optional parameters
    // so I can keep defaults in OdometrySystem::Config directly
    template<typename T>
    static std::optional<T> parameter(rclcpp::Node* node, const std::string& name, std::optional<T> to = std::nullopt) {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
        rcl_interfaces::msg::ParameterDescriptor desc{};
        desc.dynamic_typing = true;
        if constexpr (std::is_integral_v<T>) {
            rcl_interfaces::msg::IntegerRange range{};
            range.from_value = 0;
            range.to_value = to ? static_cast<rcl_interfaces::msg::IntegerRange::_to_value_type>(*to) : \
                std::numeric_limits<rcl_interfaces::msg::IntegerRange::_to_value_type>::max();
            desc.integer_range.push_back(range);
        } else if constexpr (std::is_floating_point_v<T>) {
            rcl_interfaces::msg::FloatingPointRange range{};
            range.from_value = 0.0;
            range.to_value = to ? static_cast<rcl_interfaces::msg::FloatingPointRange::_to_value_type>(*to) : \
                std::numeric_limits<rcl_interfaces::msg::FloatingPointRange::_to_value_type>::max();
            desc.floating_point_range.push_back(range);
        }
        node->declare_parameter(name, rclcpp::ParameterValue{}, desc);
        T value;
        if(node->get_parameter(name, value)) return value;
        return std::nullopt;
    }

    // TODO: Consider the upper bounds of these parameters
    static OdometrySystem::Config configure(rclcpp::Node* node) {
        OdometrySystem::Config config;
        // map
        auto max_distance = OdometryNode::parameter<double>(node, "max_distance");
        if(max_distance) config.map.max_distance = *max_distance;
        auto voxel_size = OdometryNode::parameter<double>(node, "voxel_size");
        if(voxel_size) config.map.voxel_size = *voxel_size;
        auto voxel_capacity = OdometryNode::parameter<size_t>(node, "voxel_capacity");
        if(voxel_capacity) config.map.voxel_capacity = *voxel_capacity;
        // threshold
        auto threshold_initial = OdometryNode::parameter<double>(node, "threshold_initial");
        if(threshold_initial) config.threshold.threshold_initial = *threshold_initial;
        auto min_motion = OdometryNode::parameter<double>(node, "min_motion");
        if(min_motion) config.threshold.min_motion = *min_motion;
        // registration
        auto max_iterations = OdometryNode::parameter<size_t>(node, "max_iterations");
        if(max_iterations) config.registration.max_iterations = *max_iterations;
        auto max_threads = OdometryNode::parameter<size_t>(node, "max_threads");
        if(max_threads) config.registration.max_threads = *max_threads;
        auto convergence_criterion = OdometryNode::parameter<double>(node, "convergence_criterion");
        if(convergence_criterion) config.registration.convergence_criterion = *convergence_criterion;
        auto correspondence_multiplier = OdometryNode::parameter<double>(node, "correspondence_distance_multiplier");
        if(correspondence_multiplier) config.registration.correspondence_multiplier = *correspondence_multiplier;
        return config;
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OdometryNode>();
    rclcpp::spin(std::static_pointer_cast<rclcpp::Node>(node));
    rclcpp::shutdown();
    return 0;
}