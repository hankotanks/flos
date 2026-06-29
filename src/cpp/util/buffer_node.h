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
    explicit BufferNode(const std::string& incoming_topic_name, const std::string& incoming_topic_default = "");
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
    // start worker thread
    void start() noexcept;
    // manual stop
    void stop() noexcept;
    // queue operations
    void push(const std::shared_ptr<Message>& msg) noexcept;
    std::optional<Entry> pop() noexcept;
    // pure virtual function for converting `Message` to `Entry`
    virtual Entry read(const std::shared_ptr<Message>& msg) noexcept = 0;
    // `worker` to be dispatched to a separate thread
    virtual void loop() noexcept = 0;
protected:
    // total capacity of the buffer
    static constexpr size_t CAPACITY{15};
    // whether or not the capacity should be respected
    bool debug_;
private:
    rclcpp::Subscription<Message>::SharedPtr incoming_subscriber_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<Entry> buffer_;
    std::atomic<bool> running_{true};
    std::thread thread_;
};

template<typename Entry, typename Message>
BufferNode<Entry, Message>::BufferNode(const std::string& incoming_topic_name, const std::string& incoming_topic_default) : rclcpp::Node(NODE_NAME) {
    // declare parameters
    this->declare_parameter<bool>("debug", false);
    if(incoming_topic_default.empty()) {
        this->declare_parameter<std::string>(incoming_topic_name);
    } else {
        this->declare_parameter<std::string>(incoming_topic_name, incoming_topic_default);
    }
    // store parameters
    this->get_parameter("debug", debug_);
    this->get_parameter(incoming_topic_name, incoming_topic_);
    // create subscriber
    incoming_subscriber_ = this->create_subscription<Message>(incoming_topic_, 10,
        [this](const std::shared_ptr<Message> msg) { this->push(msg); });
}

template<typename Entry, typename Message>
void BufferNode<Entry, Message>::start() noexcept {
    // initialize thread
    thread_ = std::thread(&BufferNode::loop, this);
}

// NOTE: this MUST be invoked in the destructor of all derived nodes
template<typename Entry, typename Message>
void BufferNode<Entry, Message>::stop() noexcept {
    {
        std::lock_guard lock(mutex_);
        running_ = false;
    }
    cv_.notify_all();
    // close the worker
    if(thread_.joinable()) thread_.join();
}

template<typename Entry, typename Message>
BufferNode<Entry, Message>::~BufferNode() {
    BufferNode::stop();
}

template<typename Entry, typename Message>
void BufferNode<Entry, Message>::push(const std::shared_ptr<Message>& msg) noexcept {
    if(msg->width * msg->height == 0) return;
    { // ensure capacity isn't reached
        std::unique_lock lock(mutex_);
        if(buffer_.size() >= BufferNode::CAPACITY) {
            RCLCPP_WARN_ONCE(this->get_logger(), NODE_NAME " can't keep up!");
            if(!debug_) return;
        }
        // construct and push the entry
        buffer_.emplace(std::move(this->read(msg)));
        cv_.notify_one();
    }   
}

template<typename Entry, typename Message>
std::optional<Entry> BufferNode<Entry, Message>::pop() noexcept {
    std::unique_lock lock(mutex_);

    cv_.wait(lock, [&] {
        return !buffer_.empty() || !running_ || !rclcpp::ok();
    });

    if (buffer_.empty())
        return std::nullopt;

    auto entry = std::move(buffer_.front());
    buffer_.pop();
    return entry;
}
