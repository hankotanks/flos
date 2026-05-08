#pragma once

// system
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <optional>
// ros
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
// other
#include "util/cloud.h"

class BufferNode : public rclcpp::Node {
public:
    BufferNode();
    BufferNode(const BufferNode&) = delete;
    BufferNode(BufferNode&&) = delete;
    BufferNode& operator=(const BufferNode&) = delete;
    BufferNode& operator=(BufferNode&&) = delete;
    ~BufferNode();
public:
    using Entry = std::pair<util::Cloud, std_msgs::msg::Header>;
protected:
    // parameters
    std::string incoming_points_topic;
    std::optional<double> voxel_size;
protected:
    size_t size() const noexcept { return buffer_.size(); };
    // queue operations
    void push(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) noexcept;
    std::optional<Entry> pop() noexcept;
    // block until the buffer is empty and there is no ongoing segmentation
    void wait_until_drained() noexcept;
protected:
    // total capacity of the buffer
    static constexpr size_t capacity = 5;
    // nodes deriving from this class should set this when actively processing a buffer entry
    std::atomic<bool> active{false};
private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr incoming_points_subscriber_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<Entry> buffer_;
    std::atomic<bool> running_{true};
};
