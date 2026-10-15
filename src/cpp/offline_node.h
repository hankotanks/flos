#pragma once

// system
#include <filesystem>
#include <fstream>
#include <iomanip>
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
#include "odometry/time_system.h"
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
        segmentation_system_(segmentation_config_.config, this),
        odometry_config_(this),
        odometry_system_(odometry_config_.config, this) {
        if(!rclcpp::ok()) return;
        util::param_opt_path(this, "path_to_result", path_to_result_, false, false, true);     
        util::param_opt(this, "frames", msg_count_);
        util::param_opt(this, "visualize", visualize_);
        util::param_opt(this, "visualize_map", visualize_map_);
        // configure the output for the profiler
        if(!path_to_result_.empty()) util::Profiler::save(path_to_result_ / "profiler.txt");
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
            TimeSystem::Stamp stamp(header.stamp.sec, header.stamp.nanosec);
            Sophus::SE3f pose;
            PROFILE(frame, this) {
                // preprocess before inference
                model.preprocess(cloud);
                // inference
                model.infer(cloud);
                // compute pose and store for OfflineNode::save
                pose = odometry_system_.register_to_map(cloud, stamp);
            }
            output_.emplace_back(header, pose, odometry_system_.statistics_);
            // publications
            if(visualize_) {
                SemanticCloud cloud_wrapper{cloud};
                cloud_wrapper.transform(pose);
                this->publish_scan(stamp, cloud);
                if(visualize_map_) {
                    util::Cloud out;
                    std::vector<uint8_t> prediction;
                    for(const auto& [point, semantics] : odometry_system_.points()) {
                        out.add_point(point);
                        prediction.emplace_back(semantics.prediction);
                    }
                    out.add_field<uint8_t>("prediction", prediction);
                    this->publish_local_map(stamp, out);
                    this->publish_local_map_belief(stamp, odometry_system_.map_.voxels(), odometry_config_.config.voxel_size);
                }
                this->publish_pose(stamp, header.frame_id, pose);
                this->publish_ground_plane(stamp, odometry_system_.projection_.ground_.centroid_, odometry_system_.projection_.ground_.n_, odometry_config_.config.max_distance_ground);
                if(projection_target_) this->publish_projection(stamp, *projection_target_, odometry_system_.projection_.target_);
                projection_target_.emplace(odometry_system_.projection_.target_);
            }
            const auto [percent, remaining] = OfflineNode::compute_progress();
            this->publish_progress(stamp, percent, remaining);
            this->publish_statistics(stamp, odometry_system_.statistics_);
        }
    }
protected:
    virtual void publish_scan(TimeSystem::Stamp stamp, const util::Cloud& cloud) { /* STUB */ }
    virtual void publish_local_map(TimeSystem::Stamp stamp, const util::Cloud& cloud) { /* STUB */ };
    virtual void publish_local_map_belief(TimeSystem::Stamp stamp, const VoxelMap<3, SemanticCloud>::view_voxels& voxels, float voxel_size) { /* STUB */ }
    virtual void publish_pose(TimeSystem::Stamp stamp, const std_msgs::msg::Header::_frame_id_type& frame_id, const Sophus::SE3f& pose) { /* STUB */ }
    virtual void publish_ground_plane(TimeSystem::Stamp stamp, const Eigen::Vector3f& centroid, const Eigen::Vector3f& normal, float radius) { /* STUB */ }
    virtual void publish_projection(TimeSystem::Stamp stamp, const VoxelMap<2, BirdsEyeCloud>& previous, const VoxelMap<2, BirdsEyeCloud>& current) { /* STUB */ }
    virtual void publish_progress(TimeSystem::Stamp stamp, float percent, float remaining) { /* STUB */ }
    virtual void publish_statistics(TimeSystem::Stamp stamp, const OdometrySystem::Statistics& statistics) { /* STUB */ }
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
        if(output_.empty()) return;
        if(path_to_result_.empty()) return;
        // record TUM poses
        std::ofstream f_pose{path_to_result_ / "output_tum.txt"};
        std::ofstream f_stat{path_to_result_ / "statistics.csv"};
        for(auto& [header, pose, statistics] : output_) {
            auto t = pose.translation();
            Eigen::Quaterniond q(pose.so3().unit_quaternion());
            f_pose << std::fixed << std::setprecision(9)
                << (header.stamp.sec + header.stamp.nanosec * 1e-9) << " "
                << std::setprecision(6)
                << t.x() << " " << t.y() << " " << t.z() << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                << std::endl;
            f_stat << std::fixed << std::setprecision(9)
                << (header.stamp.sec + header.stamp.nanosec * 1e-9) << ','
                << std::setprecision(6) 
                << statistics.align.iterations << ',' << statistics.align.mean_correspondence_count << ','
                << statistics.align.chi << ',' << statistics.align.refined << ',' 
                << statistics.velocity.linear << ',' << statistics.velocity.angular
                << std::endl;
        }
        std::ofstream f_para{path_to_result_ / "parameters.txt"};
        f_para << segmentation_config_.config << std::endl;
        f_para << odometry_config_.config << std::endl;
        RCLCPP_INFO(this->get_logger(), "saved results to %s [%lu poses]", path_to_result_.c_str(), output_.size());
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
    std::vector<std::tuple<std_msgs::msg::Header, Sophus::SE3f, OdometrySystem::Statistics>> output_;
};
