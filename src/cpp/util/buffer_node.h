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
    explicit BufferNode(const rclcpp::NodeOptions& options, const std::string& incoming_topic_name, const std::string& incoming_topic_default);
    BufferNode(const BufferNode&) = delete;
    BufferNode(BufferNode&&) = delete;
    BufferNode& operator=(const BufferNode&) = delete;
    BufferNode& operator=(BufferNode&&) = delete;
    ~BufferNode();
protected:
    // parameters
    std::string incoming_topic;
protected:
    // return size of the buffer
    size_t size() const noexcept { return buffer_.size(); };
    // queue operations
    void push(const std::shared_ptr<Message>& msg) noexcept;
    std::optional<Entry> pop() noexcept;
    // block until the buffer is empty and there is no ongoing segmentation
    void wait_until_drained() noexcept;
    // pure virtual function for converting `Message` to `Entry`
    virtual Entry read(const std::shared_ptr<Message>& msg) const noexcept = 0;
protected:
    // total capacity of the buffer
    static constexpr size_t capacity = 5;
    // nodes deriving from this class should set this when actively processing a buffer entry
    std::atomic<bool> active{false};
private:
    rclcpp::Subscription<Message>::SharedPtr incoming_subscriber_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<Entry> buffer_;
    std::atomic<bool> running_{true};
};

template<typename Entry, typename Message>
BufferNode<Entry, Message>::BufferNode(const rclcpp::NodeOptions& options,
    const std::string& incoming_topic_name, 
    const std::string& incoming_topic_default) : rclcpp::Node(PROJECT_NAME, options) {
    // declare parameters
    this->declare_parameter<std::string>(incoming_topic_name, incoming_topic_default);
    // store parameters
    this->get_parameter(incoming_topic_name, incoming_topic);
    // create subscriber
    incoming_subscriber_ = this->create_subscription<Message>(incoming_topic, 10,
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
        if(buffer_.size() >= capacity) {
            RCLCPP_WARN_ONCE(this->get_logger(), "segmentation_node can't keep up!");
            return;
        }
        // construct and push the entry
        buffer_.emplace(std::move(this->read(msg)));
        cv_.notify_one();
    }   
}

template<typename Entry, typename Message>
std::optional<Entry> BufferNode<Entry, Message>::pop() noexcept {
    { // wait
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return !running_ || !buffer_.empty(); });
        if(!running_ && buffer_.empty()) return std::nullopt; // exhausted
    }
    // pop points from buffer and return
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
            if(buffer_.empty() && !active.load()) break;
        }
        rate.sleep();
    }
}
