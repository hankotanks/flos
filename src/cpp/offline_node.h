#pragma once

// system
#include <filesystem>
#include <fstream>
// ros
#include <limits>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
// package
#include "odometry/system.h"
#include "segmentation/system.h"
// util
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/header.hpp"
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
// path_to_result [REQUIRED]
// path_to_model
// frames

// TOPICS
// sub: incoming_points_topic [PointCloud2]
// pub: $/bag_progress [Float32MultiArray]
// pub: $/odometry [Odometry]
// pub: $/map [PointCloud2]
// pub: $/map/pre [PointCloud2]
// pub: $/map/points [PointCloud2]
// pub: /tf [TF]

class OfflineNode : public rclcpp::Node, util::Profiler {
public:
    explicit OfflineNode() : rclcpp::Node(NODE_NAME), util::Profiler(NODE_NAME),
        config_(this),
        segmentation_system_(config_.segmentation),
        odometry_system_(config_.odometry) {
        // declare parameters
        this->declare_parameter<std::string>("incoming_points_topic");
        this->declare_parameter<std::string>("path_to_bag");
        this->declare_parameter<std::string>("path_to_result");
        // store parameters
        util::opt_param<size_t>(this, msg_count_, "frames");
        this->get_parameter("incoming_points_topic", incoming_points_topic_);
        // path of bag to read
        std::string path_to_bag;
        this->get_parameter("path_to_bag", path_to_bag);
        if(path_to_bag.empty()) {
            RCLCPP_ERROR_ONCE(this->get_logger(), "path_to_bag is required");
            rclcpp::shutdown();
        } else if(!std::filesystem::exists(path_to_bag)) {
            RCLCPP_ERROR_STREAM_ONCE(this->get_logger(), "path_to_bag does not exist: " << path_to_bag);
            rclcpp::shutdown();
        }
        path_to_bag_ = path_to_bag;
        // path for output
        std::string path_to_result;
        this->get_parameter("path_to_result", path_to_result);
        if(path_to_result.empty()) {
            RCLCPP_ERROR_ONCE(this->get_logger(), "path_to_result is required");
            rclcpp::shutdown();
        }
        path_to_result_ = path_to_result;
        // create output directory (if it doesn't exist)
        std::filesystem::path path_to_result_parent = path_to_result_.parent_path();
        if(!std::filesystem::exists(path_to_result_parent)) {
            std::filesystem::create_directories(path_to_result_parent);
        }
        // path to model
        std::filesystem::path path_to_model_default{PROJECT_ROOT};
        while(path_to_model_default.filename() != PROJECT_NAME) 
            path_to_model_default = path_to_model_default.parent_path();
        path_to_model_default /= "models";
        path_to_model_default /= "default.ckpt";
        // declare topics and model path parameters
        this->declare_parameter<std::string>("path_to_model", path_to_model_default.string());
        // store the parameters
        this->get_parameter("path_to_model", path_to_model_);
        // ensure that model path exists
        if(path_to_model_.empty()) path_to_model_ = path_to_model_default;
        if(!std::filesystem::exists(path_to_model_))
            throw std::runtime_error("model does not exist: " + path_to_model_);
        
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
private:
    struct Config {
        SegmentationSystem::Config segmentation;
        OdometrySystem::Config odometry;
        // TODO: the code duplication from segmentation_node and odometry_node should be avoided
        Config(rclcpp::Node* node) {
            // segmentation system
            util::opt_param<float>(node, segmentation.voxel_size, "voxel_size");
            util::opt_param<float>(node, segmentation.max_distance, "max_distance");
            // TODO: I would like my system to be class-name agnostic, so I've moved these parameters here
            // however, the actual class numbers should not be hard-coded...
            // this is a problem for a different day
            bool ignore_ground = segmentation.ignore.count(1) > 0;
            bool ignore_trunks = segmentation.ignore.count(3) > 0;
            bool ignore_canopy = segmentation.ignore.count(4) > 0;
            bool ignore_shrubs = segmentation.ignore.count(2) > 0;
            util::opt_param<bool>(node, ignore_ground, "ignore_ground");
            util::opt_param<bool>(node, ignore_trunks, "ignore_trunks");
            util::opt_param<bool>(node, ignore_canopy, "ignore_canopy");
            util::opt_param<bool>(node, ignore_shrubs, "ignore_shrubs");
            ignore_ground ? (segmentation.ignore.insert(1), 0) : segmentation.ignore.erase(1);
            ignore_trunks ? (segmentation.ignore.insert(3), 0) : segmentation.ignore.erase(3);
            ignore_canopy ? (segmentation.ignore.insert(4), 0) : segmentation.ignore.erase(4);
            ignore_shrubs ? (segmentation.ignore.insert(2), 0) : segmentation.ignore.erase(2);
            // log parameters
            RCLCPP_INFO_STREAM(node->get_logger(), 
                std::boolalpha << "SegmentationSystem::Config" << std::endl <<
                    "\tvoxel_size: " << segmentation.voxel_size << "," << std::endl << 
                    "\tmax_distance: " << segmentation.max_distance << "," << std::endl << 
                    "\tignore_ground: " << ignore_ground << "," << std::endl << 
                    "\tignore_trunks: " << ignore_trunks << "," << std::endl << 
                    "\tignore_canopy: " << ignore_canopy << "," << std::endl << 
                    "\tignore_shrubs: " << ignore_shrubs);
            // odometry system
            odometry.map.max_distance = segmentation.max_distance;
            odometry.map.voxel_size = segmentation.voxel_size;
            util::opt_param<size_t>(node, odometry.map.voxel_capacity, "voxel_capacity");
            // threshold
            util::opt_param<float>(node, odometry.threshold.threshold_initial, "threshold_initial");
            util::opt_param<float>(node, odometry.threshold.min_motion, "min_motion");
            // registration
            util::opt_param<bool>(node, odometry.registration.confidence_weighting, "confidence_weighting");
            util::opt_param<bool>(node, odometry.registration.correspondence_restriction, "correspondence_restriction");
            util::opt_param<bool>(node, odometry.registration.correspondence_with_voxel_belief, "correspondence_with_voxel_belief");
            util::opt_param<size_t>(node, odometry.registration.max_iterations, "max_iterations");
            util::opt_param<size_t>(node, odometry.registration.max_threads, "max_threads");
            util::opt_param<float>(node, odometry.registration.convergence_criterion, "convergence_criterion");
            util::opt_param<float>(node, odometry.registration.correspondence_multiplier, "correspondence_distance_multiplier");
            // log parameters
            RCLCPP_INFO_STREAM(node->get_logger(), 
                std::boolalpha << "OdometrySystem::Config" << std::endl <<
                    "\tmax_distance: " << odometry.map.max_distance << "," << std::endl <<
                    "\tvoxel_size: " << odometry.map.voxel_size << "," << std::endl <<
                    "\tvoxel_capacity: " << odometry.map.voxel_capacity << "," << std::endl <<
                    "\tthreshold_initial: " << odometry.threshold.threshold_initial << "," << std::endl <<
                    "\tmin_motion: " << odometry.threshold.min_motion << "," << std::endl <<
                    "\tconfidence_weighting: " << odometry.registration.confidence_weighting << "," << std::endl <<
                    "\tcorrespondence_restriction: " << odometry.registration.correspondence_restriction << "," << std::endl <<
                    "\tcorrespondence_with_voxel_belief: " << odometry.registration.correspondence_with_voxel_belief << "," << std::endl <<
                    "\tmax_iterations: " << odometry.registration.max_iterations << "," << std::endl <<
                    "\tmax_threads: " << odometry.registration.max_threads << "," << std::endl <<
                    "\tconvergence_criterion: " << odometry.registration.convergence_criterion << "," << std::endl <<
                    "\tcorrespondence_multiplier: " << odometry.registration.correspondence_multiplier);
        }
    };
public:
    void run() {
        py::scoped_interpreter guard{};
        SegmentationSystem::Model model = segmentation_system_.load(path_to_model_);
        RCLCPP_INFO_ONCE(this->get_logger(), NODE_NAME " successfully loaded inference model");
        // construct bag reader
        bag_ = std::make_unique<util::BufferableBag>(path_to_bag_, 
            // NOTE: this REQUIRES OfflineSegmentationNode to be constructed with std::make_shared
            std::make_shared<util::BufferableBag::TFBridge>(this->shared_from_this()), 
            std::vector<std::string>{incoming_points_topic_});
        // set total message count
        if(msg_count_ == 0) msg_count_ = bag_->message_count();
        // infer lidar messages from the given bag
        while(rclcpp::ok() && !bag_->finished() && msg_processed_++ < msg_count_) {
            // get next message from bag
            const rosbag2_storage::SerializedBagMessage serialized_bag_msg = bag_->PopNextMessage();
            const auto& topic = serialized_bag_msg.topic_name;
            const rclcpp::SerializedMessage msg_ser(*serialized_bag_msg.serialized_data);
            // this condition should always be truthy... but whatever...
            if(topic == incoming_points_topic_) {
                // deserialize the incoming message
                std::shared_ptr<sensor_msgs::msg::PointCloud2> msg = deserialize_next_msg<sensor_msgs::msg::PointCloud2>(msg_ser);
                // convert cloud and preprocess for inference model
                util::Cloud cloud{msg};
                Sophus::SE3f pose;
                {
                    util::Profiler profiler{"frame", this};
                    // inference
                    model.infer(cloud, false);
                    // compute pose and store for OfflineNode::save
                    pose = odometry_system_.register_to_map(cloud, msg->header.stamp.sec, msg->header.stamp.nanosec);
                }
                poses_.emplace_back(msg->header, pose);
                // publications
                this->publish_scan(msg->header.stamp, cloud);
                if(local_map_) {
                    this->publish_local_map(msg->header.stamp, *local_map_, true);
                }
                OfflineNode::update_local_map();
                this->publish_local_map(msg->header.stamp, *local_map_);
                this->publish_pose(msg->header.stamp, msg->header.frame_id, pose);
                this->publish_statistics(msg->header.stamp, odometry_system_.statistics);
                const auto [percent, remaining] = OfflineNode::compute_bag_progress();
                this->publish_bag_progress(msg->header.stamp, percent, remaining);
            }
        }
        // close bag
        bag_->close();
        bag_ = nullptr;
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
    std::string path_to_model_;
private:
    // config
    Config config_;
    // system
    SegmentationSystem segmentation_system_;
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
