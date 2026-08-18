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
#include "odometry/ground_plane.h"
#include "odometry/semantic_cloud.h"
#include "segmentation_node_config.h"
#include "segmentation/system.h"
#include "odometry_node_config.h"
#include "odometry/system.h"
#include "dataloader/dataloader.h"
// util
#include "util/cloud.h"
#include "util/rosarg.h"

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

class OfflineNode : public rclcpp::Node, util::Profiler {
public:
    explicit OfflineNode() : rclcpp::Node(NODE_NAME), util::Profiler(NODE_NAME),
        loader_(Dataloader::choose(this)),
        segmentation_config_(this),
        segmentation_system_(segmentation_config_.config),
        odometry_config_(this),
        odometry_system_(odometry_config_.config) {
        if(!rclcpp::ok()) return;
        util::param_opt_path(this, "path_to_result", path_to_result_, false, true, true);     
        util::param_opt(this, "frames", msg_count_);
        util::param_opt(this, "visualize", visualize_);
        util::param_opt(this, "visualize_map", visualize_map_);
    }
    // delete extraneous constructors
    OfflineNode(const OfflineNode&) = delete;
    OfflineNode(OfflineNode&&) = delete;
    OfflineNode& operator=(const OfflineNode&) = delete;
    OfflineNode& operator=(OfflineNode&&) = delete;
    ~OfflineNode() {
        OfflineNode::save();
    }
public:
    void run() {
        if(!rclcpp::ok()) return;
        py::scoped_interpreter guard{};
        SegmentationSystem::Model model = segmentation_system_.load(segmentation_config_.path_to_model);
        RCLCPP_INFO_ONCE(this->get_logger(), NODE_NAME " successfully loaded inference model");
        // initialize and set total message count
        size_t frames = this->loader_->init(this->shared_from_this());
        if(msg_count_ == 0) msg_count_ = frames;
        // infer lidar messages from the given bag
        while(rclcpp::ok() && !this->loader_->finished() && msg_processed_++ < msg_count_) {
            auto [cloud, header] = this->loader_->next();
            Sophus::SE3f pose;
            PROFILE(frame, this) {
                // preprocess before inference
                model.preprocess(cloud);
                // inference
                model.infer(cloud);
                // compute pose and store for OfflineNode::save
                pose = odometry_system_.register_to_map(cloud, header.stamp.sec, header.stamp.nanosec);
            }
            poses_.emplace_back(header, pose);
            // publications
            if(visualize_) {
                SemanticCloud cloud_wrapper{cloud};
                cloud_wrapper.transform(pose);
                this->publish_scan(header.stamp, cloud);
                // TODO: removed visualize_map, it needs to be added back
                if(visualize_map_) {
                    util::Cloud out;
                    std::vector<uint8_t> prediction;
                    prediction.reserve(odometry_system_.map_.size() * odometry_system_.config_.voxel_capacity);
                    for(const auto& [point, semantics] : odometry_system_.map_.view()) {
                        out.add_point(point);
                        prediction.emplace_back(semantics.prediction);
                    }
                    out.add_field<uint8_t>("prediction", prediction);
                    this->publish_local_map(header.stamp, out);
                }
                this->publish_pose(header.stamp, header.frame_id, pose);
                if(projection_target_) this->publish_projection(header.stamp, *projection_target_, odometry_system_.projection_.target_);
                projection_target_.emplace(odometry_system_.projection_.target_);
            }
            this->publish_statistics(header.stamp, odometry_system_.statistics_);
            const auto [percent, remaining] = OfflineNode::compute_progress();
            this->publish_progress(header.stamp, percent, remaining);
        }
    }
public:
    using Stamp = std_msgs::msg::Header::_stamp_type;
protected:
    virtual void publish_scan(const Stamp& stamp, const util::Cloud& cloud) { /* STUB */ };
    virtual void publish_local_map(const Stamp& stamp, const util::Cloud& cloud) { /* STUB */ };
    virtual void publish_pose(const Stamp& stamp, const std_msgs::msg::Header::_frame_id_type& frame_id, const Sophus::SE3f& pose) { /* STUB */ };
    virtual void publish_projection(const Stamp& stamp, const VoxelMap<2, BirdsEyeCloud>& previous, const VoxelMap<2, BirdsEyeCloud>& current) { /* STUB */ };
    virtual void publish_statistics(const Stamp& stamp, const OdometrySystem::Statistics& statistics) { /* STUB */ };
    virtual void publish_progress(const Stamp& stamp, float percent, float remaining) { /* STUB */ }
private:
    std::pair<float, float> compute_progress() {
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
protected:
    bool visualize_{true};
    bool visualize_map_{true};
private:
    // parameters
    std::filesystem::path path_to_result_;
    // dataloader
    std::unique_ptr<Dataloader> loader_{nullptr};
    // segmentation
    SegmentationNodeConfig segmentation_config_;
    SegmentationSystem segmentation_system_;
    // odometry
    OdometryNodeConfig odometry_config_;
    OdometrySystem odometry_system_;
    // buffer projection to show alignment
    std::optional<VoxelMap<2, BirdsEyeCloud>> projection_target_;
    // progress
    size_t msg_count_{0};
    size_t msg_processed_{0};
    // pose data
    std::vector<std::pair<std_msgs::msg::Header, Sophus::SE3f>> poses_;
};
