#pragma once

// system
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>
#include <optional>
// eigen
#include <Eigen/Core>
// ros
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
// other
#include "util/cloud.h"

namespace util {
class BaseNode : public rclcpp::Node {
public:
    BaseNode();
    BaseNode(const BaseNode&) = delete;
    BaseNode(BaseNode&&) = delete;
    BaseNode& operator=(const BaseNode&) = delete;
    BaseNode& operator=(BaseNode&&) = delete;
    ~BaseNode() {
        buffer.stop();
        if(segmentation_thread.joinable()) segmentation_thread.join();
    }
protected:
    // parameters
    std::string incoming_points_topic;
    std::string outgoing_points_topic;
    std::string path_to_model;
protected:
    struct Buffer {
        static constexpr size_t capacity = 5;
        void stop() noexcept;
        void push(rclcpp::Logger logger, const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) noexcept;
        std::optional<std::pair<util::Cloud, std_msgs::msg::Header>> pop() noexcept;
        size_t size() const noexcept { return buffer_.size(); };
        void wait_until_empty(const std::atomic<bool>& segmentation_active) noexcept;
    private:
        std::mutex mutex_;
        std::condition_variable cv_;
        std::queue<std::pair<util::Cloud, std_msgs::msg::Header>> buffer_;
        std::atomic<bool> running_{true};
    };
protected:
    // topic stuff  
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr incoming_points_subscriber;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr outgoing_points_publisher;
    // buffer for incoming point data
    Buffer buffer;
    // separate thread for processing buffered clouds
    std::thread segmentation_thread;
    // signals when inference is underway
    std::atomic<bool> segmentation_active{false};
protected:
    void loop();
    virtual void infer(py::object infer, util::Cloud& cloud, const std_msgs::msg::Header& header) noexcept;
    void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) noexcept { 
        buffer.push(this->get_logger(), msg);
    }
};
}
