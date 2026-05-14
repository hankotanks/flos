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
    explicit SegmentationNode();
    SegmentationNode(const SegmentationNode&) = delete;
    SegmentationNode(SegmentationNode&&) = delete;
    SegmentationNode& operator=(const SegmentationNode&) = delete;
    SegmentationNode& operator=(SegmentationNode&&) = delete;
    ~SegmentationNode() {
        if(segmentation_thread.joinable()) segmentation_thread.join();
    }
protected:
    // parameters
    std::string outgoing_points_topic;
    std::string path_to_model;
    std::optional<double> voxel_size;
protected:
    // topic stuff  
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr outgoing_points_publisher;
    // separate thread for processing buffered clouds
    std::thread segmentation_thread;
protected:
    void loop();
    virtual void infer(py::object infer, util::Cloud& cloud) const;
    // overload of the BufferNode::read method
    virtual std::pair<util::Cloud, std_msgs::msg::Header> read(const std::shared_ptr<sensor_msgs::msg::PointCloud2>& msg) noexcept final override;
};

