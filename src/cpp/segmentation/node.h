#pragma once

// system
#include <thread>
// eigen
#include <Eigen/Core>
// ros
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
// util
#include "util/buffer_node.h"
#include "util/cloud.h"

class SegmentationNode : public BufferNode<std::pair<util::Cloud, std_msgs::msg::Header>, sensor_msgs::msg::PointCloud2> {
public:
    struct Config {
        double voxel_size{1.0};
        double max_distance{25.0};
        bool ignore_canopy{true};
        bool ignore_shrubs{true};
    };
public:
    explicit SegmentationNode();
    SegmentationNode(const SegmentationNode&) = delete;
    SegmentationNode(SegmentationNode&&) = delete;
    SegmentationNode& operator=(const SegmentationNode&) = delete;
    SegmentationNode& operator=(SegmentationNode&&) = delete;
    ~SegmentationNode() {
        if(segmentation_thread_.joinable()) segmentation_thread_.join();
    }
protected:
    // parameters
    std::string outgoing_points_topic_;
    std::string path_to_model_;
protected:
    Config config_;
    // scalar for voxel_size
    // NOTE: Config::voxel_size is scaled by this constant for unified parameter passing between segmentation and odometry
    static constexpr double VOXEL_SIZE_SCALAR{0.5};
    // scalar for map distance
    // NOTE: the idea is that we want the map to be larger than any incoming scan's max distance
    static constexpr double MAX_DISTANCE_SCALAR{0.75};
    // topic stuff  
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr outgoing_points_publisher_;
    // separate thread for processing buffered clouds
    std::thread segmentation_thread_;
protected:
    void loop();
    virtual void infer(py::object infer, util::Cloud& cloud);
    // overload of the BufferNode::read method
    virtual std::pair<util::Cloud, std_msgs::msg::Header> read(const std::shared_ptr<sensor_msgs::msg::PointCloud2>& msg) noexcept final override;
private:
    static SegmentationNode::Config configure(rclcpp::Node* node);
};

