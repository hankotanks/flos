// system
#include <limits>
// ros
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
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
        // store parameters
        this->get_parameter("local_map_topic", local_map_topic);
        this->get_parameter("local_map_frame", local_map_frame);
        // initialize map publisher
        local_map_publisher = this->create_publisher<sensor_msgs::msg::PointCloud2>(local_map_topic, 10);  
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
protected:
    static constexpr double local_map_rate = 5.0; // number of seconds between map publications
    // core
    OdometrySystem system;
    // publisher for local map
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_publisher;
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
            // perform registration
            // TODO: Need to think about whether this should be the original cloud's timestamp or overwritten with this->now()
            auto [cloud, header] = std::move(*maybe);
            const Sophus::SE3f& pose = system.register_to_map(std::move(cloud));
            { // send the transformation from the lidar frame to the local map frame
                geometry_msgs::msg::TransformStamped msg;
                msg.header.stamp = header.stamp;
                msg.header.frame_id = header.frame_id;
                msg.child_frame_id = local_map_frame;
                // transform position
                msg.transform.translation.x = pose.translation().x();
                msg.transform.translation.y = pose.translation().y();
                msg.transform.translation.z = pose.translation().z();
                // tranform heading
                Eigen::Quaternionf q(pose.so3().unit_quaternion());
                msg.transform.rotation.x = q.x();
                msg.transform.rotation.y = q.y();
                msg.transform.rotation.z = q.z();
                msg.transform.rotation.w = q.w();
                // publish the transform
                tf_broadcaster->sendTransform(msg);
            }
            // send the local map
            if((this->now() - local_map_last_publication).seconds() >= OdometryNode::local_map_rate) {
                header.frame_id = local_map_frame;
                const util::CloudView& cloud = system.cloud();
                local_map_publisher->publish(std::move(cloud.message(header)));
                local_map_last_publication = this->now();
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
    static std::optional<T> parameter(rclcpp::Node* node, const std::string& name, T to) {
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
    static OdometrySystem::Config configure(rclcpp::Node* node) {
        OdometrySystem::Config config;
        // map
        auto max_distance = OdometryNode::parameter(node, "max_distance", std::numeric_limits<double>::max());
        if(max_distance) config.map.max_distance = *max_distance;
        auto voxel_size = OdometryNode::parameter(node, "voxel_size", std::numeric_limits<double>::max());
        if(voxel_size) config.map.voxel_size = *voxel_size;
        auto voxel_capacity = OdometryNode::parameter(node, "voxel_capacity", std::numeric_limits<size_t>::max());
        if(voxel_capacity) config.map.voxel_capacity = *voxel_capacity;
        // threshold
        auto threshold_initial = OdometryNode::parameter(node, "threshold_initial", std::numeric_limits<double>::max());
        if(threshold_initial) config.threshold.threshold_initial = *threshold_initial;
        auto min_motion = OdometryNode::parameter(node, "min_motion", std::numeric_limits<double>::max());
        if(min_motion) config.threshold.min_motion = *min_motion;
        // registration
        auto max_iterations = OdometryNode::parameter(node, "max_iterations", std::numeric_limits<size_t>::max());
        if(max_iterations) config.registration.max_iterations = *max_iterations;
        auto max_threads = OdometryNode::parameter(node, "max_threads", std::numeric_limits<size_t>::max());
        if(max_threads) config.registration.max_threads = *max_threads;
        auto convergence_criterion = OdometryNode::parameter(node, "convergence_criterion", std::numeric_limits<double>::max());
        if(convergence_criterion) config.registration.convergence_criterion = *convergence_criterion;
        auto correspondence_multiplier = OdometryNode::parameter(node, "correspondence_distance_multiplier", std::numeric_limits<size_t>::max());
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