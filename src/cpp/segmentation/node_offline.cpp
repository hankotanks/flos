#include <stdexcept>
#include <filesystem>
// ros
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
// package
#include "nav_msgs/msg/odometry.hpp"
#include "node.h"
// util
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "util/rosbag.h"
#include "util/profiler.h"
#include "util/delayed_publisher.h"

namespace {
template <typename T>
std::shared_ptr<T> deserialize_next_msg(const rclcpp::SerializedMessage& msg_ser) {
    const auto msg = std::make_shared<T>();
    const rclcpp::Serialization<T> serializer;
    serializer.deserialize_message(&msg_ser, msg.get());
    return msg;
}
} // namespace

class OfflineSegmentationNode : util::Profiler, public SegmentationNode {
public:
    explicit OfflineSegmentationNode() : util::Profiler(NODE_NAME) {
        // declare and store parameters
        this->declare_parameter<std::string>("local_map_frame", "odom_frame"); // NOTE: this must match the equivalent OdometryNode parameter
        this->declare_parameter<std::string>("path_to_bag");
        this->declare_parameter<std::string>("imu_topic", "");
        std::string local_map_frame;
        this->get_parameter("local_map_frame", local_map_frame);
        this->get_parameter("path_to_bag", path_to_bag_);
        // validate path_to_bag
        if(path_to_bag_.empty()) {
            throw std::runtime_error("path_to_bag is required");
        } else if(!std::filesystem::exists(path_to_bag_)) {
            throw std::runtime_error("path_to_bag does not exist: " + path_to_bag_);
        }
        this->get_parameter("imu_topic", imu_topic_);
        // publish statistics
        bag_progress_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/" PROJECT_NAME "/bag_progress", 10);
        // initialize DelayedPublisher
        delayed_publisher_.initialize(this, "/" PROJECT_NAME "/odometry");
    }
    // delete extraneous constructors
    OfflineSegmentationNode(const OfflineSegmentationNode&) = delete;
    OfflineSegmentationNode(OfflineSegmentationNode&&) = delete;
    OfflineSegmentationNode& operator=(const OfflineSegmentationNode&) = delete;
    OfflineSegmentationNode& operator=(OfflineSegmentationNode&&) = delete;
    ~OfflineSegmentationNode() {
        delayed_publisher_.stop();
    }
public:
    void run() {
        // construct bag reader
        bag_ = std::make_unique<util::BufferableBag>(path_to_bag_, 
            // NOTE: this REQUIRES OfflineSegmentationNode to be constructed with std::make_shared
            std::make_shared<util::BufferableBag::TFBridge>(this->shared_from_this()), 
            std::vector{incoming_topic_, imu_topic_});
        // set total message count
        msg_count_ = bag_->message_count();
        // infer lidar messages from the given bag
        rclcpp::Rate rate(10);
        while(rclcpp::ok() && !bag_->finished()) {
            if(BufferNode::size() >= BUFFER_THRESHOLD) {
                RCLCPP_WARN_STREAM_ONCE(this->get_logger(), "throttling bag read " << 
                    "[buffer size: " << BufferNode::size() << ", capacity: " << BufferNode::CAPACITY << "]");
                rate.sleep();
                continue;
            }
            // get next message from bag
            const rosbag2_storage::SerializedBagMessage serialized_bag_msg = bag_->PopNextMessage();
            const auto& topic = serialized_bag_msg.topic_name;
            const rclcpp::SerializedMessage msg_ser(*serialized_bag_msg.serialized_data);
            // this condition should always be truthy... but whatever...
            if(topic == incoming_topic_) {
                std::shared_ptr<sensor_msgs::msg::PointCloud2> msg = deserialize_next_msg<sensor_msgs::msg::PointCloud2>(msg_ser);
                BufferNode::push(msg);
                delayed_publisher_.queue(msg);
            } else if(topic == imu_topic_) {
                std::shared_ptr<sensor_msgs::msg::Imu> msg = deserialize_next_msg<sensor_msgs::msg::Imu>(msg_ser);
                delayed_publisher_.queue(msg);
            }
            // increment processed message count
            msg_processed_++;
            // publish updated progress
            OfflineSegmentationNode::publish_bag_progress();
        }
        // wait until the remaining scans are consumed by the segmentation loop
        BufferNode::wait_until_drained();
        // clean up DelayedPublisher
        delayed_publisher_.stop();
    }
private:
    void publish_bag_progress() {
        if(msg_count_ == 0) return;

        static const auto start_time = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        const float elapsed_seconds = std::chrono::duration<float>(now - start_time).count();

        const float percent_complete = 100.f * static_cast<float>(msg_processed_) / static_cast<float>(msg_count_);
        const float avg_time_per_msg = (msg_processed_ > 0) ? elapsed_seconds / static_cast<float>(msg_processed_) : 0.f;
        const float seconds_remaining = avg_time_per_msg * static_cast<float>(msg_count_ - msg_processed_);

        std_msgs::msg::Float32MultiArray msg;
        msg.data.resize(2);
        msg.data[0] = percent_complete;
        msg.data[1] = seconds_remaining;

        bag_progress_publisher_->publish(msg);
    }

    void infer(py::object infer, util::Cloud& cloud) final override {
        util::Profiler* this_profiler = this;
        util::Profiler profiler{"inference", this_profiler};
        SegmentationNode::infer(infer, cloud);
    }
private:
    // parameters
    std::string path_to_bag_;
    std::string imu_topic_;
private:
    
private:
    // used for stalling the bag reader when the buffer is mostly full
    static constexpr size_t BUFFER_THRESHOLD{static_cast<size_t>(0.9 * static_cast<double>(BufferNode::CAPACITY))};
    // bag
    std::unique_ptr<util::BufferableBag> bag_;
    size_t msg_count_{0};
    size_t msg_processed_{0};
    // publish the bags progress (including estimated time to completion)
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr bag_progress_publisher_;
    // delayed publisher for raw point data
    util::DelayedPublisher<nav_msgs::msg::Odometry> delayed_publisher_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OfflineSegmentationNode>();
    // TODO: OfflineSegmentationNode should launch this thread
    std::thread bag_thread([node] { node->run(); });
    rclcpp::spin(node);
    bag_thread.join();
    rclcpp::shutdown();
    return 0;
}