#pragma once

// system
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
// ros
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
// package
#include "segmentation_node_config.h"
#include "segmentation/system.h"
#include "odometry_node_config.h"
#include "odometry/system.h"
// util
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "util/cloud.h"
#include "util/rosbag.h"
#include "util/rosargs.h"

namespace {
template <typename T>
std::shared_ptr<T> deserialize_next_msg(const rclcpp::SerializedMessage& msg_ser) {
    const auto msg = std::make_shared<T>();
    const rclcpp::Serialization<T> serializer;
    serializer.deserialize_message(&msg_ser, msg.get());
    return msg;
}
} // namespace

// PARAMETERS
// incoming_points_topic [REQUIRED]
// path_to_bag [REQUIRED]
// path_to_result
// path_to_model
// frames

// TOPICS
// sub: incoming_points_topic [PointCloud2]

class OfflineNode : public rclcpp::Node, util::Profiler {
public:
    explicit OfflineNode(const bool flags = true) : rclcpp::Node(NODE_NAME), util::Profiler(NODE_NAME),
        segmentation_config_(this),
        segmentation_system_(segmentation_config_.config),
        odometry_config_(this),
        odometry_system_(odometry_config_.config) {
        // declare parameters
        if(flags) {
            util::param_req(this, "incoming_points_topic", incoming_points_topic_);
            util::param_req_path(this, "path_to_bag", path_to_bag_, true, true);
        }
        util::param_opt_path(this, "path_to_result", path_to_result_, true, false, true);        
    }
    // delete extraneous constructors
    OfflineNode(const OfflineNode&) = delete;
    OfflineNode(OfflineNode&&) = delete;
    OfflineNode& operator=(const OfflineNode&) = delete;
    OfflineNode& operator=(OfflineNode&&) = delete;
    ~OfflineNode() {
        if(bag_) {
            bag_->close();
        }
        OfflineNode::save();
    }
public:
    void run() {
        py::scoped_interpreter guard{};
        SegmentationSystem::Model model = segmentation_system_.load(segmentation_config_.path_to_model);
        RCLCPP_INFO_ONCE(this->get_logger(), NODE_NAME " successfully loaded inference model");
        // initialize and set total message count
        if(msg_count_ == 0) msg_count_ = this->init();
        // infer lidar messages from the given bag
        while(rclcpp::ok() && !this->finished() && msg_processed_++ < msg_count_) {
            auto [cloud, header] = this->next();
            Sophus::SE3f pose;
            {
                util::Profiler profiler{"frame", this};
                // inference
                model.infer(cloud, false);
                // compute pose and store for OfflineNode::save
                pose = odometry_system_.register_to_map(cloud, header.stamp.sec, header.stamp.nanosec);
            }
            poses_.emplace_back(header, pose);
            // publications
            this->publish_scan(header.stamp, cloud);
            if(local_map_) {
                this->publish_local_map(header.stamp, *local_map_, true);
            }
            OfflineNode::update_local_map();
            this->publish_local_map(header.stamp, *local_map_);
            this->publish_pose(header.stamp, header.frame_id, pose);
            this->publish_statistics(header.stamp, odometry_system_.statistics);
            const auto [percent, remaining] = OfflineNode::compute_bag_progress();
            this->publish_bag_progress(header.stamp, percent, remaining);
        }
        
        // close bag
        bag_->close();
        bag_ = nullptr;
    }
protected:
    virtual size_t init() {
        // construct bag reader
        bag_ = std::make_unique<util::BufferableBag>(path_to_bag_, 
            // NOTE: this REQUIRES OfflineSegmentationNode to be constructed with std::make_shared
            std::make_shared<util::BufferableBag::TFBridge>(this->shared_from_this()), 
            std::vector<std::string>{incoming_points_topic_});

        return bag_->message_count();
    }

    virtual bool finished() {
        return bag_->finished();
    }

    virtual std::pair<util::Cloud, std_msgs::msg::Header> next() {
        // get next message from bag
        const rosbag2_storage::SerializedBagMessage serialized_bag_msg = bag_->PopNextMessage();
        const auto& topic = serialized_bag_msg.topic_name;
        const rclcpp::SerializedMessage msg_ser(*serialized_bag_msg.serialized_data);
        // this condition should always be truthy... but whatever...
        std::shared_ptr<sensor_msgs::msg::PointCloud2> msg = deserialize_next_msg<sensor_msgs::msg::PointCloud2>(msg_ser);
        return std::make_pair(msg, msg->header);
    }
protected:
    virtual void publish_scan(const std_msgs::msg::Header::_stamp_type& stamp, const util::CloudView& cloud) { /* STUB */ };
    virtual void publish_local_map(const std_msgs::msg::Header::_stamp_type& stamp, const util::CloudView& cloud, bool previous = false) { /* STUB */ };
    virtual void publish_pose(const std_msgs::msg::Header::_stamp_type& stamp, const std_msgs::msg::Header::_frame_id_type& frame_id, const Sophus::SE3f& pose) { /* STUB */ };
    virtual void publish_statistics(const std_msgs::msg::Header::_stamp_type& stamp, const OdometrySystem::Statistics& statistics) { /* STUB */ };
    virtual void publish_bag_progress(const std_msgs::msg::Header::_stamp_type& stamp, float percent, float remaining) { /* STUB */ }
private:
    std::pair<float, float> compute_bag_progress() {
        if(msg_count_ == 0) return std::make_pair(0.f, std::numeric_limits<float>::max());
        static const auto start_time = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        const float elapsed_seconds = std::chrono::duration<float>(now - start_time).count();
        const float percent_complete = 100.f * static_cast<float>(msg_processed_) / static_cast<float>(msg_count_);
        const float avg_time_per_msg = (msg_processed_ > 0) ? elapsed_seconds / static_cast<float>(msg_processed_) : 0.f;
        const float seconds_remaining = avg_time_per_msg * static_cast<float>(msg_count_ - msg_processed_);

        return std::make_pair(percent_complete, seconds_remaining);
    }

    void save() {
        if(poses_.empty()) return;
        if(path_to_result_.empty()) return;
        // record TUM poses
        std::ofstream f{path_to_result_};
        for(auto& [header, pose] : poses_) {
            auto t = pose.translation();
            Eigen::Quaterniond q(pose.so3().unit_quaternion());
            f << std::fixed << std::setprecision(9)
                << (header.stamp.sec + header.stamp.nanosec * 1e-9) << " "
                << std::setprecision(6)
                << t.x() << " " << t.y() << " " << t.z() << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                << std::endl;
        }
        RCLCPP_INFO(this->get_logger(), "saved %s [%lu poses]", path_to_result_.c_str(), poses_.size());
    }

    void update_local_map() {
        const auto& local_map = odometry_system_.cloud();
        local_map_ = std::make_unique<util::Cloud>(local_map.points);
        { // TODO: lots of copying going on here, should be improved
            static std::vector<uint8_t> semantics;
            semantics.clear();
            semantics.reserve(local_map.data.size());
            std::transform(local_map.data.begin(), local_map.data.end(), std::back_inserter(semantics), 
                [](const Semantics& p) { return p.prediction; });
            local_map_->add_field<uint8_t>("class", semantics);
        }
    }
private:
    // parameters
    std::string incoming_points_topic_;
    std::filesystem::path path_to_bag_;
    std::filesystem::path path_to_result_;
private:
    // segmentation
    SegmentationNodeConfig segmentation_config_;
    SegmentationSystem segmentation_system_;
    // odometry
    OdometryNodeConfig odometry_config_;
    OdometrySystem odometry_system_;
    // bag
    std::unique_ptr<util::BufferableBag> bag_;
    size_t msg_count_{0};
    size_t msg_processed_{0};
    // local map
    std::unique_ptr<util::Cloud> local_map_{nullptr};
    // pose data
    std::vector<std::pair<std_msgs::msg::Header, Sophus::SE3f>> poses_;
};
