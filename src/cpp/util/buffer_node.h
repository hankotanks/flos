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

template<typename Entry, typename Message>
class BufferNode : public rclcpp::Node {
public:
    BufferNode() = delete;
    explicit BufferNode(const std::string& incoming_topic_name, const std::string& incoming_topic_default);
    BufferNode(const BufferNode&) = delete;
    BufferNode(BufferNode&&) = delete;
    BufferNode& operator=(const BufferNode&) = delete;
    BufferNode& operator=(BufferNode&&) = delete;
    ~BufferNode();
protected:
    // parameters
    std::string incoming_topic_;
protected:
    // return size of the buffer
    size_t size() const noexcept { return buffer_.size(); };
    // queue operations
    void push(const std::shared_ptr<Message>& msg) noexcept;
    std::optional<Entry> pop() noexcept;
    // block until the buffer is empty and there is no ongoing segmentation
    void wait_until_drained() noexcept;
    // pure virtual function for converting `Message` to `Entry`
    virtual Entry read(const std::shared_ptr<Message>& msg) noexcept = 0;
protected:
    // total capacity of the buffer
    static constexpr size_t CAPACITY{15};
    // nodes deriving from this class should set this when actively processing a buffer entry
    std::atomic<bool> active_{false};
private:
    rclcpp::Subscription<Message>::SharedPtr incoming_subscriber_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<Entry> buffer_;
    std::atomic<bool> running_{true};
};

template<typename Entry, typename Message>
BufferNode<Entry, Message>::BufferNode(const std::string& incoming_topic_name, const std::string& incoming_topic_default) : rclcpp::Node(NODE_NAME) {
    // declare parameters
    this->declare_parameter<std::string>(incoming_topic_name, incoming_topic_default);
    // store parameters
    this->get_parameter(incoming_topic_name, incoming_topic_);
    // create subscriber
    incoming_subscriber_ = this->create_subscription<Message>(incoming_topic_, 10,
        [this](const std::shared_ptr<Message> msg) { this->push(msg); });
}

template<typename Entry, typename Message>
BufferNode<Entry, Message>::~BufferNode() {
    { // stop
        std::lock_guard lock(mutex_);
        running_ = false; 
    }
    cv_.notify_all(); 
}

template<typename Entry, typename Message>
void BufferNode<Entry, Message>::push(const std::shared_ptr<Message>& msg) noexcept {
    if(msg->width * msg->height == 0) return;
    { // ensure capacity isn't reached
        std::unique_lock lock(mutex_);
        if(buffer_.size() >= CAPACITY) {
            RCLCPP_WARN_ONCE(this->get_logger(), NODE_NAME " can't keep up!");
            return;
        }
        // construct and push the entry
        buffer_.emplace(std::move(this->read(msg)));
        cv_.notify_one();
    }   
}

template<typename Entry, typename Message>
std::optional<Entry> BufferNode<Entry, Message>::pop() noexcept {
    std::unique_lock lock(mutex_);
    while(buffer_.empty() && running_ && rclcpp::ok()) { // NOTE: I've add rclcpp::ok here for cleaner shutdown sequences
        cv_.wait_for(lock, std::chrono::milliseconds(50));
    }
    if(!running_ && buffer_.empty()) return std::nullopt;
    if(buffer_.empty()) return std::nullopt;
    auto entry = std::move(buffer_.front());
    buffer_.pop();
    return entry;
}

template<typename Entry, typename Message>
void BufferNode<Entry, Message>::wait_until_drained() noexcept {
    rclcpp::Rate rate(100);
    while(rclcpp::ok()) {
        { // wait for buffer to drain
            std::lock_guard<std::mutex> lock(mutex_);
            if(buffer_.empty() && !active_.load()) break;
        }
        rate.sleep();
    }
}
