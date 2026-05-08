#pragma once

// system
#include <cassert>
#include <thread>
// eigen
#include <Eigen/Core>
// ros
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
// forest_segmentation
#include "buffer_node.h"
// util
#include "util/cloud.h"

class BaseNode : public BufferNode {
public:
    BaseNode();
    BaseNode(const BaseNode&) = delete;
    BaseNode(BaseNode&&) = delete;
    BaseNode& operator=(const BaseNode&) = delete;
    BaseNode& operator=(BaseNode&&) = delete;
    ~BaseNode() {
        if(segmentation_thread.joinable()) segmentation_thread.join();
    }
protected:
    // parameters
    std::string outgoing_points_topic;
    std::string path_to_model;
protected:
    // topic stuff  
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr outgoing_points_publisher;
    // separate thread for processing buffered clouds
    std::thread segmentation_thread;
protected:
    void loop();
    virtual void infer(py::object infer, util::Cloud& cloud, const std_msgs::msg::Header& header) noexcept;
};

