#include <stdexcept>
#include <filesystem>
// ros
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
// other
#include "util/base_node.h"
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

class OfflineSegmentationNode : public util::BaseNode, util::Profiler {
public:
    OfflineSegmentationNode() : util::BaseNode(), util::Profiler(std::filesystem::path(__FILE__).stem()) {
        // declare and store parameters
        this->declare_parameter<std::string>("path_to_bag");
        this->get_parameter("path_to_bag", path_to_bag);
        // validate path_to_bag
        if(path_to_bag.empty()) {
            throw std::runtime_error("path_to_bag is required");
        } else if(!std::filesystem::exists(path_to_bag)) {
            throw std::runtime_error("path_to_bag does not exist: " + path_to_bag);
        }
        // publish statistics
        bag_progress_publisher = this->create_publisher<std_msgs::msg::Float32MultiArray>("/" PROJECT_NAME "/bag_progress", 10);
    }
    // delete extraneous constructors
    OfflineSegmentationNode(const OfflineSegmentationNode&) = delete;
    OfflineSegmentationNode(OfflineSegmentationNode&&) = delete;
    OfflineSegmentationNode& operator=(const OfflineSegmentationNode&) = delete;
    OfflineSegmentationNode& operator=(OfflineSegmentationNode&&) = delete;
public:
    void run() {
        // construct bag reader
        bag = std::make_unique<util::BufferableBag>(path_to_bag, 
            // NOTE: this REQUIRES OfflineSegmentationNode to be constructed with std::make_shared
            std::make_shared<util::BufferableBag::TFBridge>(this->shared_from_this()), 
            std::vector{incoming_points_topic});
        // set total message count
        msg_count = bag->message_count();
        // infer lidar messages from the given bag
        while(rclcpp::ok() && !bag->finished()) {
            if(buffer.size() >= buffer_threshold) {
                RCLCPP_WARN_STREAM_ONCE(this->get_logger(), "throttling bag read [buffer size: " << buffer.size()
                    << ", capacity: " << Buffer::capacity << "]");
                // TODO: better method for throttling
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            // get next message from bag
            const rosbag2_storage::SerializedBagMessage serialized_bag_msg = bag->PopNextMessage();
            const auto& topic = serialized_bag_msg.topic_name;
            const rclcpp::SerializedMessage msg_ser(*serialized_bag_msg.serialized_data);
            // this condition should always be truthy... but whatever...
            if(topic == incoming_points_topic) {
                util::BaseNode::callback(deserialize_next_msg<sensor_msgs::msg::PointCloud2>(msg_ser));
            }
            // increment processed message count
            msg_processed++;
            // publish updated progress
            OfflineSegmentationNode::publish_bag_progress();
        }
        // wait until the remaining scans are consumed by the segmentation loop
        buffer.wait_until_empty(segmentation_active);
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

    void infer(py::object infer, util::Cloud& cloud, const std_msgs::msg::Header& header) noexcept final override {
        util::Profiler* this_profiler = this;
        util::Profiler profiler("inference", this_profiler);
        BaseNode::infer(infer, cloud, header);
    }
private:
    static constexpr size_t buffer_threshold = static_cast<size_t>(0.9 * static_cast<double>(Buffer::capacity));
    std::string path_to_bag;
    std::unique_ptr<util::BufferableBag> bag;
    size_t msg_count = 0;
    size_t msg_processed = 0;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr bag_progress_publisher;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OfflineSegmentationNode>();
    node->run();
    rclcpp::shutdown();
    return 0;
}