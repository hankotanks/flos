#include "offline_node.h"

// ros
#include <std_msgs/msg/float32_multi_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

// PARAMETERS
// path_to_data [REQUIRED]
// path_to_result
// path_to_model
// frames
// visualize
// visualize_map
// incoming_points_topic [REQUIRED for BagDataloader]

// TOPICS
// sub: incoming_points_topic [PointCloud2]
// pub: $/progress [Float32MultiArray]
// pub: $/odometry [Odometry]
// pub: $/map [PointCloud2]
// pub: $/scan [PointCloud2]
// pub: /tf [TF]

class DefaultOfflineNode : public OfflineNode {
public:
    explicit DefaultOfflineNode() : OfflineNode() {
        if(!rclcpp::ok()) return;
        scan_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/" PROJECT_NAME "/map/points", 10);
        local_map_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/" PROJECT_NAME "/map", 10);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/" PROJECT_NAME "/odometry", 10);
        progress_publisher = this->create_publisher<std_msgs::msg::Float32MultiArray>("/" PROJECT_NAME "/progress", 10);
    }
protected:
    virtual void publish_scan(TimeSystem::Stamp stamp, const util::Cloud& cloud) override {
        std_msgs::msg::Header header;
        header.stamp.sec = stamp.sec;
        header.stamp.nanosec = stamp.nanosec;
        header.frame_id = DefaultOfflineNode::LOCAL_MAP_FRAME;
        scan_publisher_->publish(std::move(cloud.message(header)));
    }

    virtual void publish_local_map(TimeSystem::Stamp stamp, const util::Cloud& cloud) override {
        std_msgs::msg::Header header;
        header.stamp.sec = stamp.sec;
        header.stamp.nanosec = stamp.nanosec;
        header.frame_id = DefaultOfflineNode::LOCAL_MAP_FRAME;
        local_map_publisher_->publish(std::move(cloud.message(header)));
    }

    virtual void publish_pose(TimeSystem::Stamp stamp, const std_msgs::msg::Header::_frame_id_type& frame_id, const Sophus::SE3f& pose) override {
        {
            geometry_msgs::msg::TransformStamped msg;
            msg.header.stamp.sec = stamp.sec;
            msg.header.stamp.nanosec = stamp.nanosec;
            msg.header.frame_id = DefaultOfflineNode::LOCAL_MAP_FRAME;
            msg.child_frame_id = frame_id;
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
        {
            nav_msgs::msg::Odometry msg;
            msg.header.stamp.sec = stamp.sec;
            msg.header.stamp.nanosec = stamp.nanosec;
            msg.header.frame_id = DefaultOfflineNode::LOCAL_MAP_FRAME;
            msg.child_frame_id = frame_id;
            const auto t = pose.translation();
            msg.pose.pose.position.x = t.x();
            msg.pose.pose.position.y = t.y();
            msg.pose.pose.position.z = t.z();
            Eigen::Quaternionf q(pose.so3().unit_quaternion());
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
            odometry_publisher_->publish(std::move(msg));
        }
    }

    virtual void publish_progress(TimeSystem::Stamp stamp, float percent, float remaining) override {
        std_msgs::msg::Float32MultiArray msg;
        msg.data.resize(2);
        msg.data[0] = percent;
        msg.data[1] = remaining;
        progress_publisher->publish(msg);
    }
private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_publisher_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr progress_publisher;
private:
    // output frame_id
    static constexpr std::string LOCAL_MAP_FRAME{"odom_frame"};
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DefaultOfflineNode>();
    node->run();
    rclcpp::shutdown();
    return 0;
}
