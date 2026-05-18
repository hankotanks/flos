#include <stdexcept>
#include <filesystem>
// ros
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
// package
#include "node.h"
// util
#include "util/rosbag.h"
#include "util/profiler.h"

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
        std::string local_map_frame;
        this->get_parameter("local_map_frame", local_map_frame);
        this->get_parameter("path_to_bag", path_to_bag);
        // validate path_to_bag
        if(path_to_bag.empty()) {
            throw std::runtime_error("path_to_bag is required");
        } else if(!std::filesystem::exists(path_to_bag)) {
            throw std::runtime_error("path_to_bag does not exist: " + path_to_bag);
        }
        // publish statistics
        bag_progress_publisher = this->create_publisher<std_msgs::msg::Float32MultiArray>("/" PROJECT_NAME "/bag_progress", 10);
        // initialize DelayedPublisher
        delayed_publisher.initialize(this, "/" PROJECT_NAME "/raw", local_map_frame);
    }
    // delete extraneous constructors
    OfflineSegmentationNode(const OfflineSegmentationNode&) = delete;
    OfflineSegmentationNode(OfflineSegmentationNode&&) = delete;
    OfflineSegmentationNode& operator=(const OfflineSegmentationNode&) = delete;
    OfflineSegmentationNode& operator=(OfflineSegmentationNode&&) = delete;
    ~OfflineSegmentationNode() {
        delayed_publisher.stop();
    }
public:
    void run() {
        // construct bag reader
        bag = std::make_unique<util::BufferableBag>(path_to_bag, 
            // NOTE: this REQUIRES OfflineSegmentationNode to be constructed with std::make_shared
            std::make_shared<util::BufferableBag::TFBridge>(this->shared_from_this()), 
            std::vector{incoming_topic});
        // set total message count
        msg_count = bag->message_count();
        // infer lidar messages from the given bag
        rclcpp::Rate rate(10);
        while(rclcpp::ok() && !bag->finished()) {
            if(BufferNode::size() >= buffer_threshold) {
                RCLCPP_WARN_STREAM_ONCE(this->get_logger(), "throttling bag read " << 
                    "[buffer size: " << BufferNode::size() << ", capacity: " << BufferNode::capacity << "]");
                rate.sleep();
                continue;
            }
            // get next message from bag
            const rosbag2_storage::SerializedBagMessage serialized_bag_msg = bag->PopNextMessage();
            const auto& topic = serialized_bag_msg.topic_name;
            const rclcpp::SerializedMessage msg_ser(*serialized_bag_msg.serialized_data);
            // this condition should always be truthy... but whatever...
            if(topic == incoming_topic) {
                std::shared_ptr<sensor_msgs::msg::PointCloud2> msg = deserialize_next_msg<sensor_msgs::msg::PointCloud2>(msg_ser);
                BufferNode::push(msg);
                delayed_publisher.queue(msg);
            }
            // increment processed message count
            msg_processed++;
            // publish updated progress
            OfflineSegmentationNode::publish_bag_progress();
        }
        // wait until the remaining scans are consumed by the segmentation loop
        BufferNode::wait_until_drained();
        // clean up DelayedPublisher
        delayed_publisher.stop();
    }
private:
    void publish_bag_progress() {
        if(msg_count == 0) return;

        static const auto start_time = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        const float elapsed_seconds = std::chrono::duration<float>(now - start_time).count();

        const float percent_complete = 100.f * static_cast<float>(msg_processed) / static_cast<float>(msg_count);
        const float avg_time_per_msg = (msg_processed > 0) ? elapsed_seconds / static_cast<float>(msg_processed) : 0.f;
        const float seconds_remaining = avg_time_per_msg * static_cast<float>(msg_count - msg_processed);

        std_msgs::msg::Float32MultiArray msg;
        msg.data.resize(2);
        msg.data[0] = percent_complete;
        msg.data[1] = seconds_remaining;

        bag_progress_publisher->publish(msg);
    }

    void infer(py::object infer, util::Cloud& cloud) final override {
        util::Profiler* this_profiler = this;
        util::Profiler profiler{"inference", this_profiler};
        SegmentationNode::infer(infer, cloud);
    }
private:
    // parameters
    std::string path_to_bag;
private:
    class DelayedPublisher {
    public:
        DelayedPublisher() = default;
        DelayedPublisher(const DelayedPublisher&) = delete;
        DelayedPublisher(DelayedPublisher&&) = delete;
        DelayedPublisher& operator=(const DelayedPublisher&) = delete;
        DelayedPublisher& operator=(DelayedPublisher&&) = delete;
        ~DelayedPublisher() = default;
    public:
        void initialize(rclcpp::Node* node, const std::string& topic_name, const std::string& local_map_frame) {
            logger_ = node->get_logger();
            local_map_frame_ = local_map_frame;
            // initialize TF2
            tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node->get_clock());
            tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
            // publish original points (will be published by dedicated thread)
            publisher_ = node->create_publisher<sensor_msgs::msg::PointCloud2>(topic_name, 10);
            // launch publication thread
            thread_ = std::thread(&DelayedPublisher::publish, this);
        }

        // terminate the publisher
        void stop() {
            { // signal thread to terminate
                std::unique_lock<std::mutex> lock(mutex_);
                running_ = false;
            }
            cv_.notify_all();
            if(thread_.joinable()) thread_.join();
        }

        // queue a scan
        void queue(const std::shared_ptr<sensor_msgs::msg::PointCloud2>& msg) {
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
            while(running_) {
                std::shared_ptr<sensor_msgs::msg::PointCloud2> msg;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this] { return !queue_.empty() || !running_; });
                    if(!running_ && queue_.empty()) break;
                    if(queue_.empty()) continue;
                    msg = queue_.front();
                    queue_.pop();
                }
                try {
                    // TODO: it's unacceptable that "odom_frame" is specified here as a literal
                    tf_buffer_->lookupTransform(local_map_frame_, 
                        msg->header.frame_id, 
                        msg->header.stamp, 
                        tf2::durationFromSec(10.0));
                    publisher_->publish(*msg);
                } catch(const tf2::TransformException& ex) {
                    if(logger_) {
                        RCLCPP_WARN_STREAM_ONCE(*logger_, NODE_NAME << " didn't find transformations to " << 
                            local_map_frame_ << ", halting publication of raw points");
                    }
                    std::unique_lock<std::mutex> lock(mutex_);
                    running_ = false;
                }
            }
        }
    private:
        std::optional<rclcpp::Logger> logger_;
        std::string local_map_frame_;
        std::queue<std::shared_ptr<sensor_msgs::msg::PointCloud2>> queue_;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
        std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
        std::thread thread_;
        std::atomic<bool> running_{true};
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
        
    };
private:
    // used for stalling the bag reader when the buffer is mostly full
    static constexpr size_t buffer_threshold = static_cast<size_t>(0.9 * static_cast<double>(BufferNode::capacity));
    // bag
    std::unique_ptr<util::BufferableBag> bag;
    size_t msg_count = 0;
    size_t msg_processed = 0;
    // publish the bags progress (including estimated time to completion)
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr bag_progress_publisher;
    // delayed publisher for raw point data
    DelayedPublisher delayed_publisher;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OfflineSegmentationNode>();
    node->run();
    rclcpp::shutdown();
    return 0;
}