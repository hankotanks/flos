#include <atomic>
#include <filesystem>
#include <fstream>
// ros
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
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

int64_t ns_from_header(const std_msgs::msg::Header& header) {
    const auto& stamp = header.stamp;
    return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}
} // namespace

// PARAMETERS
// incoming_points_topic [REQUIRED]
// incoming_imu_topic
// path_to_bag [REQUIRED]
// path_to_result [REQUIRED]

// TOPICS
// sub: $/odometry [Odometry]
// pub: $/raw/points [PointCloud2]
// pub: $/raw/imu [Imu] (if incoming_imu_topic is provided)
// pub: $/bag_progress [Float32MultiArray]

class OfflineNode : util::Profiler, public rclcpp::Node {
public:
    explicit OfflineNode() : util::Profiler(NODE_NAME), rclcpp::Node(NODE_NAME) {
        // declare parameters
        this->declare_parameter<std::string>("incoming_points_topic");
        this->declare_parameter<std::string>("incoming_imu_topic", "");
        this->declare_parameter<std::string>("path_to_bag");
        this->declare_parameter<std::string>("path_to_result");
        // store parameters
        this->get_parameter("incoming_points_topic", incoming_points_topic_);
        this->get_parameter("incoming_imu_topic", incoming_imu_topic_);
        // path of bag to read
        std::string path_to_bag;
        this->get_parameter("path_to_bag", path_to_bag);
        if(path_to_bag.empty()) {
            RCLCPP_ERROR_ONCE(this->get_logger(), "path_to_bag is required");
            rclcpp::shutdown();
        } else if(!std::filesystem::exists(path_to_bag)) {
            RCLCPP_ERROR_STREAM_ONCE(this->get_logger(), "path_to_bag does not exist: " << path_to_bag);
            rclcpp::shutdown();
        }
        path_to_bag_ = path_to_bag;
        // path for output
        std::string path_to_result;
        this->get_parameter("path_to_result", path_to_result);
        if(path_to_result.empty()) {
            RCLCPP_ERROR_ONCE(this->get_logger(), "path_to_result is required");
            rclcpp::shutdown();
        }
        path_to_result_ = path_to_result;
        // create output directory (if it doesn't exist)
        if(!std::filesystem::exists(path_to_result_)) {
            std::filesystem::create_directories(path_to_result_);
        }
        // publish statistics
        bag_progress_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/" PROJECT_NAME "/bag_progress", 10);
        // initialize DelayedPublisher
        listener_ = this->create_subscription<nav_msgs::msg::Odometry>("/" PROJECT_NAME "/odometry", 10, 
            [this](const std::shared_ptr<nav_msgs::msg::Odometry> msg) {
                this->last_.store(ns_from_header(msg->header), std::memory_order_relaxed);
        });
        publisher_points_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/" PROJECT_NAME "/raw/points", 10);
        if(!incoming_imu_topic_.empty()) {
            publisher_imu_ = this->create_publisher<sensor_msgs::msg::Imu>("/" PROJECT_NAME "/raw/imu", 10);
        }
        // callback for recording new topics
        topic_scan_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&OfflineNode::scan, this));
    }
    // delete extraneous constructors
    OfflineNode(const OfflineNode&) = delete;
    OfflineNode(OfflineNode&&) = delete;
    OfflineNode& operator=(const OfflineNode&) = delete;
    OfflineNode& operator=(OfflineNode&&) = delete;
    ~OfflineNode() {
        OfflineNode::save();
    }
public:
    void wait(rclcpp::Rate& rate, const std_msgs::msg::Header& header){
        const int64_t stamp = ns_from_header(header);
        while(rclcpp::ok()) {
            const int64_t last = last_.load();  // or from subscriber callback
            if(last == 0) {
                rate.sleep();
                return;
            } else if(stamp <= last + static_cast<int64_t>(5e9) || stamp < last) {
                return;
            }
            rclcpp::spin_some(this->shared_from_this());
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void run() {
        std::vector<std::string> topics;
        topics.emplace_back(incoming_points_topic_);
        if(!incoming_imu_topic_.empty()) {
            topics.emplace_back(incoming_imu_topic_);
        }
        // construct bag reader
        bag_ = std::make_unique<util::BufferableBag>(path_to_bag_, 
            // NOTE: this REQUIRES OfflineSegmentationNode to be constructed with std::make_shared
            std::make_shared<util::BufferableBag::TFBridge>(this->shared_from_this()), topics);
        // set total message count
        msg_count_ = bag_->message_count();
        // infer lidar messages from the given bag
        rclcpp::Rate rate(10);
        while(rclcpp::ok() && !bag_->finished()) {
            // get next message from bag
            const rosbag2_storage::SerializedBagMessage serialized_bag_msg = bag_->PopNextMessage();
            const auto& topic = serialized_bag_msg.topic_name;
            const rclcpp::SerializedMessage msg_ser(*serialized_bag_msg.serialized_data);
            // this condition should always be truthy... but whatever...
            if(topic == incoming_points_topic_) {
                std::shared_ptr<sensor_msgs::msg::PointCloud2> msg = deserialize_next_msg<sensor_msgs::msg::PointCloud2>(msg_ser);
                OfflineNode::wait(rate, msg->header);
                publisher_points_->publish(*msg);
            } else if(publisher_imu_ && topic == incoming_imu_topic_) {
                std::shared_ptr<sensor_msgs::msg::Imu> msg = deserialize_next_msg<sensor_msgs::msg::Imu>(msg_ser);
                OfflineNode::wait(rate, msg->header);
                publisher_imu_->publish(*msg);
            }
            // increment processed message count
            msg_processed_++;
            // publish updated progress
            OfflineNode::publish_bag_progress();
            // spin to allow subscriptions to receive incoming messages
            rclcpp::spin_some(this->shared_from_this());
        }

        // callback for exiting when all tracked topics are closed
        topic_poll_ = this->create_wall_timer(
            std::chrono::seconds(5),
            std::bind(&OfflineNode::poll, this));
        
        while(rclcpp::ok() && !shutdown_) {
            rclcpp::spin_some(this->shared_from_this());
            rate.sleep();
        }
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
private:
    struct Pose {
        std_msgs::msg::Header header;
        geometry_msgs::msg::Point position;
        geometry_msgs::msg::Quaternion orientation;
    };
private:
    void save() {
        for(auto& [topic, poses] : poses_) {
            if(poses.empty()) continue;
            // get output filename
            std::string fname = topic;
            if(!fname.empty() && fname[0] == '/') fname.erase(0, 1);
            std::replace(fname.begin(), fname.end(), '/', '_');
            // record TUM poses
            std::ofstream f{path_to_result_ / (fname + "_tum.txt")};
            for(auto& pose : poses) {
                f << std::fixed << std::setprecision(9)
                  << (pose.header.stamp.sec + pose.header.stamp.nanosec * 1e-9) << " "
                  << std::setprecision(6)
                  << pose.position.x << " " << pose.position.y << " " << pose.position.z << " "
                  << pose.orientation.x << " " << pose.orientation.y << " " << pose.orientation.z << " " << pose.orientation.w
                  << std::endl;
            }
            RCLCPP_INFO(this->get_logger(), "saved %s [%lu poses]", topic.c_str(), poses.size());
        }
    }

    void scan() {
        for(const auto& topic : this->get_topic_names_and_types()) {
            bool is_odom = false;
            for(const auto& tp : topic.second) {
                if(tp == "nav_msgs/msg/Odometry") is_odom = true;
            }
            // add a callback for each odometry topic we encounter that isn't already subscribed
            if(is_odom && topics_.find(topic.first) == topics_.end()) {
                RCLCPP_INFO(this->get_logger(), "recording %s", topic.first.c_str());
                topics_[topic.first] = this->create_subscription<nav_msgs::msg::Odometry>(topic.first, 10,
                    [this, topic](nav_msgs::msg::Odometry::SharedPtr msg) {
                        std::lock_guard<std::mutex> lock(this->pose_mutex_);
                        if(poses_.find(topic.first) == poses_.end()) {
                            poses_.emplace(std::make_pair(topic.first, std::vector<Pose>{}));
                        }
                        poses_[topic.first].emplace_back(msg->header, msg->pose.pose.position, msg->pose.pose.orientation);
                    });
            }
        }
    }

    void poll() {
        if(shutdown_) return;
        // get the current final timestamp
        int64_t last = this->last_.load(std::memory_order_acquire);
        // determine if any odometry systems are still working
        bool working = false;
        {
            std::lock_guard<std::mutex> lock(this->pose_mutex_);
            for(const auto& [topic_name, data] : poses_) {
                std::cout << topic_name << ": " << data.size() << std::endl;
                if(data.empty()) continue;
                int64_t ns = ns_from_header(data.back().header);
                if(ns > last) continue;
                int64_t delta = last - ns;
                if(delta >= 1e+9) {
                    working = true;
                    RCLCPP_INFO_STREAM(this->get_logger(), "waiting for " << topic_name << " [" 
                        << (static_cast<double>(delta) / static_cast<double>(1e9)) << "s left]");
                    break;
                }
            }
        }
        // if everything is dead we can save and exit
        if(!working) {
            RCLCPP_INFO_ONCE(this->get_logger(), "finished recording odometry. saving and shutting down");
            shutdown_.store(true, std::memory_order_relaxed);
            topic_scan_->cancel();
            topic_poll_->cancel();
        }
    }
private:
    // parameters
    std::string incoming_points_topic_;
    std::string incoming_imu_topic_;
    std::filesystem::path path_to_bag_;
    std::filesystem::path path_to_result_;
private:
    std::atomic<bool> shutdown_{false};
    // bag
    std::unique_ptr<util::BufferableBag> bag_;
    size_t msg_count_{0};
    size_t msg_processed_{0};
    // publish the bags progress (including estimated time to completion)
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr bag_progress_publisher_;
    // listener
    typename rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr listener_;
    // last received odometry message
    std::atomic<int64_t> last_{0};
    // delayed publishers
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_points_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr publisher_imu_{nullptr};
    // repeated invocations of topic poller and exit condition checker
    rclcpp::TimerBase::SharedPtr topic_scan_;
    rclcpp::TimerBase::SharedPtr topic_poll_;
    // list of odometry topics which have been subscribed to
    std::map<std::string, rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> topics_;
    // pose data
    std::map<std::string, std::vector<Pose>> poses_;
    std::mutex pose_mutex_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OfflineNode>();
    node->run();
    rclcpp::shutdown();
    return 0;
}