// system
#include <chrono>
// ros
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
// package
#include "system.h"
// util
#include "util/buffer_node.h"
#include "util/cloud.h"
#include "util/profiler.h"
#include "util/rosargs.h"

class OdometryNode : 
    util::Profiler, 
    public BufferNode<std::pair<util::CloudView, std_msgs::msg::Header>, sensor_msgs::msg::PointCloud2> {
public:
    explicit OdometryNode() : util::Profiler(NODE_NAME),
        BufferNode("incoming_points_topic", "/" PROJECT_NAME "/inference"), 
        system_(OdometryNode::configure(this)) {
        // declare parameters
        this->declare_parameter<bool>("debug", false);
        this->declare_parameter<std::string>("local_map_topic", "/" PROJECT_NAME "/local_map");
        this->declare_parameter<std::string>("local_map_frame", "odom_frame");
        // store parameters
        this->get_parameter("debug", debug_);
        this->get_parameter("local_map_topic", local_map_topic_);
        this->get_parameter("local_map_frame", local_map_frame_);
        if(debug_) { // initialize map publisher
            local_map_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(local_map_topic_, 10);
        }
        // initialize odometry publisher
        odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/" PROJECT_NAME "/odometry", 10);
        // broadcaster for poses
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        // spin up thread for odometry
        odometry_thread_ = std::thread(&OdometryNode::loop, this);
    }
public:
    OdometryNode(const OdometryNode&) = delete;
    OdometryNode(OdometryNode&&) = delete;
    OdometryNode& operator=(const OdometryNode&) = delete;
    OdometryNode& operator=(OdometryNode&&) = delete;
    ~OdometryNode() {
        if(odometry_thread_.joinable()) odometry_thread_.join();
    }
protected:
    // parameters (excluding those related to the system config itself)
    bool debug_;
    std::string local_map_topic_;
    std::string local_map_frame_;
protected:
    static constexpr double LOCAL_MAP_RATE = 5.0; // number of seconds between map publications
    // core
    OdometrySystem system_;
    // publisher for local map
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_publisher_{nullptr};
    // odometry publisher
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    // pose broadcaster
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    // separate thread for processing
    std::thread odometry_thread_;
protected:
    void loop() {
        using steady_clock = std::chrono::steady_clock;
        using time_point = steady_clock::time_point;
        steady_clock::time_point local_map_last_publication = steady_clock::now();
        while(rclcpp::ok()) {
            // get the next batch of points
            auto maybe = BufferNode::pop();
            if(!maybe) break;
            active_ = true;
            auto [cloud, header] = std::move(*maybe);
            // perform registration
            Sophus::SE3f pose;
            {   
                util::Profiler* this_profiler = this;
                util::Profiler profiler("registration", this_profiler);
                pose = system_.register_to_map(std::move(cloud));
            }
            { // send the transformation from the lidar frame to the local map frame
                geometry_msgs::msg::TransformStamped msg;
                msg.header.stamp = header.stamp;
                msg.header.frame_id = local_map_frame_;
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
                msg.header.frame_id = local_map_frame_;
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
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - local_map_last_publication);
                if(elapsed.count() >= LOCAL_MAP_RATE * 1000) {
                    std_msgs::msg::Header local_map_header;
                    local_map_header.stamp = header.stamp;
                    local_map_header.frame_id = local_map_frame_;
                    const util::CloudView& cloud = system_.cloud();
                    local_map_publisher_->publish(std::move(cloud.message(local_map_header)));
                }
                
            }
            active_ = false;
        }
    }

    virtual std::pair<util::CloudView, std_msgs::msg::Header> read(const std::shared_ptr<sensor_msgs::msg::PointCloud2>& msg) noexcept final override {
        util::CloudView cloud{msg};
        return std::make_pair(std::move(cloud), msg->header);
    }
private:
    // TODO: Consider the upper bounds of these parameters
    static OdometrySystem::Config configure(rclcpp::Node* node) {
        OdometrySystem::Config config;
        // map
        auto confidence_based_insertion = util::numeric_param<bool>(node, "confidence_based_insertion");
        if(confidence_based_insertion) config.map.confidence_based_insertion = *confidence_based_insertion;
        auto max_distance = util::numeric_param<double>(node, "max_distance");
        if(max_distance) config.map.max_distance = *max_distance;
        auto voxel_size = util::numeric_param<double>(node, "voxel_size");
        if(voxel_size) config.map.voxel_size = *voxel_size;
        auto voxel_capacity = util::numeric_param<size_t>(node, "voxel_capacity");
        if(voxel_capacity) config.map.voxel_capacity = *voxel_capacity;
        // threshold
        auto threshold_initial = util::numeric_param<double>(node, "threshold_initial");
        if(threshold_initial) config.threshold.threshold_initial = *threshold_initial;
        auto min_motion = util::numeric_param<double>(node, "min_motion");
        if(min_motion) config.threshold.min_motion = *min_motion;
        // registration
        auto confidence_weighting = util::numeric_param<bool>(node, "confidence_weighting");
        if(confidence_weighting) config.registration.confidence_weighting = *confidence_weighting;
        auto correspondence_restriction = util::numeric_param<bool>(node, "correspondence_restriction");
        if(correspondence_restriction) config.registration.correspondence_restriction = *correspondence_restriction;
        auto max_iterations = util::numeric_param<size_t>(node, "max_iterations");
        if(max_iterations) config.registration.max_iterations = *max_iterations;
        auto max_threads = util::numeric_param<size_t>(node, "max_threads");
        if(max_threads) config.registration.max_threads = *max_threads;
        auto convergence_criterion = util::numeric_param<double>(node, "convergence_criterion");
        if(convergence_criterion) config.registration.convergence_criterion = *convergence_criterion;
        auto correspondence_multiplier = util::numeric_param<double>(node, "correspondence_distance_multiplier");
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