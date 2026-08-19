// system
#include <chrono>
// ros
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
// package
#include "odometry_node_config.h"
#include "odometry/system.h"
// util
#include "util/buffer_node.h"
#include "util/cloud.h"
#include "util/profiler.h"

// PARAMETERS
// debug
// incoming_points_topic
// confidence_based_insertion
// max_distance
// voxel_size
// voxel_capacity
// threshold_initial
// min_motion
// confidence_weighting
// correspondence_restriction
// max_iterations
// max_threads
// convergence_criterion
// correspondence_distance_multiplier

// TOPICS
// sub: incoming_points_topic (default: $/inference) [PointCloud2]
// pub: $/odometry [Odometry]
// pub: $/map [PointCloud2] (if debug is truthy)
// pub: $/map/points [PointCloud2] (if debug is truthy)
// pub: /tf [TF]

class OdometryNode : 
    util::Profiler, 
    public BufferNode<std::pair<util::Cloud, std_msgs::msg::Header>, sensor_msgs::msg::PointCloud2> {
public:
    explicit OdometryNode() : util::Profiler(NODE_NAME),
        BufferNode("incoming_points_topic", "/" PROJECT_NAME "/inference"),
        config_(this),
        system_(config_.config) {
        if(!rclcpp::ok()) return;
        if(debug_) { // initialize map publisher
            local_map_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/" PROJECT_NAME "/map", 10);
            outgoing_points_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/" PROJECT_NAME "/map/points", 10);
        }
        // initialize odometry publisher
        odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/" PROJECT_NAME "/odometry", 10);
        // broadcaster for poses
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        // start worker thread
        BufferNode::start();
    }
public:
    OdometryNode(const OdometryNode&) = delete;
    OdometryNode(OdometryNode&&) = delete;
    OdometryNode& operator=(const OdometryNode&) = delete;
    OdometryNode& operator=(OdometryNode&&) = delete;
    ~OdometryNode() { 
        BufferNode::stop(); 
    }
protected:
    // output frame_id
    static constexpr std::string LOCAL_MAP_FRAME{"odom_frame"}; 
    // number of seconds between map publications
    static constexpr float LOCAL_MAP_RATE{5.f}; 
    // config
    OdometryNodeConfig config_;
    // core
    OdometrySystem system_;
    // publisher for local map
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_publisher_{nullptr};
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr outgoing_points_publisher_{nullptr};
    // odometry publisher
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    // pose broadcaster
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
protected:
    virtual void loop() noexcept override final {
        using steady_clock = std::chrono::steady_clock;
        using time_point = steady_clock::time_point;
        steady_clock::time_point local_map_last_publication = steady_clock::now();
        while(rclcpp::ok()) {
            // get the next batch of points
            auto maybe = BufferNode::pop();
            if(!maybe) break;
            auto [cloud, header] = std::move(*maybe);
            // perform registration
            Sophus::SE3f pose;
            {
                util::Profiler profiler("registration", this);
                pose = system_.register_to_map(cloud, header.stamp.sec, header.stamp.nanosec);
            }
            { // send the transformation from the lidar frame to the local map frame
                geometry_msgs::msg::TransformStamped msg;
                msg.header.stamp = header.stamp;
                msg.header.frame_id = OdometryNode::LOCAL_MAP_FRAME;
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
                tf_broadcaster_->sendTransform(std::move(msg));
            }
            { // odometry
                nav_msgs::msg::Odometry msg;
                msg.header.stamp = header.stamp;
                msg.header.frame_id = OdometryNode::LOCAL_MAP_FRAME;
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
                odom_publisher_->publish(std::move(msg));
            }
            if(debug_) {
                // localized points publication
                outgoing_points_publisher_->publish(std::move(cloud.message(header)));
                // local map publication
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - local_map_last_publication);
                if(elapsed.count() >= OdometryNode::LOCAL_MAP_RATE * 1000) {
                    std_msgs::msg::Header local_map_header;
                    local_map_header.stamp = header.stamp;
                    local_map_header.frame_id = OdometryNode::LOCAL_MAP_FRAME;
                    util::Cloud out;
                    std::vector<uint8_t> prediction;
                    for(const auto& [point, semantics] : system_.view()) {
                        out.add_point(point);
                        prediction.emplace_back(semantics.prediction);
                    }
                    out.add_field<uint8_t>("prediction", prediction);
                    local_map_publisher_->publish(std::move(out.message(local_map_header)));
                    local_map_last_publication = steady_clock::now();
                }
            }
        }
    }

    virtual std::pair<util::Cloud, std_msgs::msg::Header> read(const std::shared_ptr<sensor_msgs::msg::PointCloud2>& msg) noexcept final override {
        util::Cloud cloud{msg};
        return std::make_pair(std::move(cloud), msg->header);
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OdometryNode>();
    rclcpp::spin(std::static_pointer_cast<rclcpp::Node>(node));
    rclcpp::shutdown();
    return 0;
}
