#pragma once

#include "sensor_msgs/msg/point_cloud2.hpp"
#include <atomic>
#include <memory>
// ros
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace {
int64_t ns_from_header(const std_msgs::msg::Header& header) {
    const auto& stamp = header.stamp;
    return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}
} // private scope

namespace util {
template<typename Listener>
class DelayedPublisher {
public:
    DelayedPublisher() = default;
    DelayedPublisher(const DelayedPublisher&) = delete;
    DelayedPublisher(DelayedPublisher&&) = delete;
    DelayedPublisher& operator=(const DelayedPublisher&) = delete;
    DelayedPublisher& operator=(DelayedPublisher&&) = delete;
    ~DelayedPublisher() = default;
public:
    using Message = std::variant<std::shared_ptr<sensor_msgs::msg::PointCloud2>, std::shared_ptr<sensor_msgs::msg::Imu>>;
public:
    void initialize(rclcpp::Node* node, const std::string& listener_topic_name) {
        initialized_ = true;
        logger_ = std::make_unique<rclcpp::Logger>(node->get_logger());
        // listener
        subscriber_ = node->create_subscription<Listener>(listener_topic_name, 10, [this](const std::shared_ptr<Listener> msg) {
            int64_t ns = ns_from_header(msg->header);
            this->last_.store(ns, std::memory_order_relaxed);
        });
        // publish original points (will be published by dedicated thread)
        publisher_lidar_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("/" PROJECT_NAME "/raw/lidar", 10);
        publisher_imu_ = node->create_publisher<sensor_msgs::msg::Imu>("/" PROJECT_NAME "/raw/imu", 10);
        // launch publication thread
        thread_ = std::thread(&DelayedPublisher::publish, this);
    }

    // terminate the publisher
    void stop() {
        if(!initialized_) return;
        { // signal thread to terminate
            std::unique_lock<std::mutex> lock(mutex_);
            running_ = false;
        }
        cv_.notify_all();
        if(thread_.joinable()) thread_.join();
    }

    // queue a scan
    void queue(const Message& msg) {
        if(!initialized_) return;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if(!running_) return;
            queue_.push(msg);
        }
        cv_.notify_one();
    }

    // runs on a separate thread to publish raw scans in concert with their transforms
    // otherwise, visualization is impossible with rviz2
    void publish() {
        if(!initialized_) return;
        while(running_) {
            std::optional<Message> msg;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return !queue_.empty() || !running_; });
                if(!running_ && queue_.empty()) break;
                if(queue_.empty()) continue;
                int64_t last = this->last_.load(std::memory_order_acquire);
                std::visit([last, &msg](const auto& msg_inner) {
                    using T = std::decay_t<decltype(msg_inner)>;
                    if constexpr (std::is_same_v<T, std::shared_ptr<sensor_msgs::msg::PointCloud2>>) {
                        if(last >= ns_from_header(msg_inner->header)) {
                            msg = msg_inner;
                        }
                    } else if constexpr (std::is_same_v<T, std::shared_ptr<sensor_msgs::msg::Imu>>) {
                        if(last >= ns_from_header(msg_inner->header)) {
                            msg = msg_inner;
                        }
                    }
                }, queue_.front());
            }
            if(msg) {
                queue_.pop();
                std::visit([this](const auto& msg_inner) {
                    using T = std::decay_t<decltype(msg_inner)>;
                    if constexpr (std::is_same_v<T, std::shared_ptr<sensor_msgs::msg::PointCloud2>>) {
                        publisher_lidar_->publish(*msg_inner);
                    } else if constexpr (std::is_same_v<T, std::shared_ptr<sensor_msgs::msg::Imu>>) {
                        publisher_imu_->publish(*msg_inner);
                    }
                }, *msg);
            }
        }
    }
private:
    bool initialized_{false};
    std::string topic_name_;
    std::unique_ptr<rclcpp::Logger> logger_;
    typename rclcpp::Subscription<Listener>::SharedPtr subscriber_;
    std::atomic<int64_t> last_{0};
    std::queue<Message> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    std::atomic<bool> running_{true};

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_lidar_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr publisher_imu_;
};
}