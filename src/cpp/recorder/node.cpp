// system
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
// ros
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>

namespace {
int64_t ns_from_header(const std_msgs::msg::Header& header) {
    const auto& stamp = header.stamp;
    return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}
} // namespace

class RecorderNode : public rclcpp::Node {
public:
    explicit RecorderNode() : rclcpp::Node(NODE_NAME) {
        // declare parameters
        this->declare_parameter<std::string>("path_to_record", "");
        // get parameters
        std::string path_to_record;
        this->get_parameter("path_to_record", path_to_record);
        if(path_to_record.empty()) {
            RCLCPP_ERROR_ONCE(this->get_logger(), "path_to_record is required");
            rclcpp::shutdown();
        }
        path_to_record_ = path_to_record;
        // create output directory (if it doesn't exist)
        if(!std::filesystem::exists(path_to_record_)) {
            std::filesystem::create_directory(path_to_record_);
        }
        // keep track of last timestamp dispatched to odometry system
        listener_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/" PROJECT_NAME "/raw/lidar", 10,
            [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                this->last_->store(ns_from_header(msg->header), std::memory_order_relaxed);
            });
        // latest time stamp of scan data
        last_ = std::make_shared<std::atomic<int64_t>>(0);
        // callback for recording new topics
        timer_scan_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&RecorderNode::scan, this));
        // callback for exiting when all tracked topics are closed
        timer_exit_ = this->create_wall_timer(
            std::chrono::seconds(5),
            std::bind(&RecorderNode::exit, this));
    }
public:
    void save() {
        for(auto& [topic, poses] : data_) {
            if(poses.empty()) continue;
            // get output filename
            std::string fname = topic;
            if(!fname.empty() && fname[0] == '/') fname.erase(0, 1);
            std::replace(fname.begin(), fname.end(), '/', '_');
            // record TUM poses
            std::ofstream f{path_to_record_ / (fname + ".tum")};
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
private:
    struct Pose {
        std_msgs::msg::Header header;
        geometry_msgs::msg::Point position;
        geometry_msgs::msg::Quaternion orientation;
    };
private:
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
                        if(data_.find(topic.first) == data_.end()) {
                            data_.emplace(std::make_pair(topic.first, std::vector<Pose>{}));
                        }
                        data_[topic.first].emplace_back(msg->header, msg->pose.pose.position, msg->pose.pose.orientation);
                    });
            }
        }
    }

    void exit() {
        if(this->topics_.empty()) return;
        if(this->listener_->get_publisher_count() > 0) return;
        // get the current final timestamp
        int64_t last = this->last_->load(std::memory_order_acquire);
        // determine if any odometry systems are still working
        bool working = false;
        {
            std::lock_guard<std::mutex> lock(this->data_mutex_);
            for(const auto& [topic_name, data] : data_) {
                if(data.empty()) continue;
                int64_t ns = ns_from_header(data.back().header);
                if(ns >= last) continue;
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
            RecorderNode::save();
            RCLCPP_INFO_ONCE(this->get_logger(), "finished recording odometry. saving and shutting down");
            rclcpp::shutdown();
        }
        
    }
private:
    std::filesystem::path path_to_record_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr listener_;
    std::shared_ptr<std::atomic<int64_t>> last_;
    rclcpp::TimerBase::SharedPtr timer_scan_;
    rclcpp::TimerBase::SharedPtr timer_exit_;
    std::map<std::string, rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> topics_;
    std::map<std::string, std::vector<Pose>> data_;
    std::mutex data_mutex_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RecorderNode>();
    rclcpp::spin(std::static_pointer_cast<rclcpp::Node>(node));
    node->save();
    rclcpp::shutdown();
    return 0;
}