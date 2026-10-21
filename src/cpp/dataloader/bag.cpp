#include "dataloader/bag.h"

// util
#include "util/rosarg.h"
#include <rosbag2_storage/topic_metadata.hpp>

namespace {
template <typename T>
std::shared_ptr<T> deserialize_next_msg(const rclcpp::SerializedMessage& msg_ser) {
    const auto msg = std::make_shared<T>();
    const rclcpp::Serialization<T> serializer;
    serializer.deserialize_message(&msg_ser, msg.get());
    return msg;
}

std::string normalize_ext(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), 
        [](unsigned char c) { return std::tolower(c); });
    return ext;
}
} // namespace

BagDataloader::BagDataloader(rclcpp::Node* node) : Dataloader(node) {
    if(std::filesystem::is_directory(path_to_data_)) {
        for(const auto& entry : std::filesystem::directory_iterator(path_to_data_)) {
            if(!entry.is_regular_file()) continue;
            const auto ext = normalize_ext(entry.path());
            if(ext != ".mcap" && ext != ".db3") continue;
            if(!path_to_bag_.empty()) throw Dataloader::UnsupportedFormat{"multiple MCAP/DB3 files in data folder"};
            path_to_bag_ = entry.path();
        }
    } else {
        const auto ext = normalize_ext(path_to_data_);
        if(ext == ".mcap" || ext == ".db3") {
            path_to_bag_ = path_to_data_;
        }
    }
    if(path_to_bag_.empty()) throw Dataloader::UnsupportedFormat{"unable to find MCAP/DB3 file"};
    // params
    util::param_opt(node, "incoming_points_topic", incoming_points_topic_);
    {
        rosbag2_cpp::Reader reader;
        reader.open(rosbag2_storage::StorageOptions{path_to_bag_.string()});
        // get pc2 topics
        std::vector<rosbag2_storage::TopicMetadata> topics = reader.get_all_topics_and_types();
        auto it = std::remove_if(topics.begin(), topics.end(), [](const rosbag2_storage::TopicMetadata& topic) { 
            return topic.type != "sensor_msgs/msg/PointCloud2"; 
        });
        topics.resize(std::distance(topics.begin(), it));
        // close reader
        reader.close();
        // handle whether incoming_points_topic is required for this bag
        if(topics.empty()) {
            RCLCPP_ERROR_STREAM_ONCE(node->get_logger(), "supplied bag has no PointCloud2 topics!");
            rclcpp::shutdown();
        } else if(!incoming_points_topic_.empty()) {
            bool valid = std::any_of(topics.begin(), topics.end(), [this](const rosbag2_storage::TopicMetadata& topic) {
                return this->incoming_points_topic_ == topic.name;
            });
            if(!valid) {
                std::ostringstream names;
                bool first = true;
                for(const auto& topic : topics) {
                    names << (first ? "" : ", ") << topic.name;
                    first = false;
                }
                RCLCPP_ERROR_STREAM_ONCE(node->get_logger(), "invalid argument, must be one of [" << names.str() << "]: incoming_points_topic");
                rclcpp::shutdown();
            }
        } else if(incoming_points_topic_.empty() && topics.size() != 1) {
            std::ostringstream names;
            bool first = true;
            for(const auto& topic : topics) {
                names << (first ? "" : ", ") << topic.name;
                first = false;
            }
            RCLCPP_ERROR_STREAM_ONCE(node->get_logger(), "missing a required argument, must be one of [" << names.str() << "]: incoming_points_topic");
            rclcpp::shutdown();
        } else if(incoming_points_topic_.empty() && topics.size() == 1) {
            incoming_points_topic_ = topics[0].name;
        }
    }
}

BagDataloader::~BagDataloader() {
    if(bag_) bag_->close();
}

size_t BagDataloader::init(const std::shared_ptr<rclcpp::Node>& node) {
    // construct bag reader
    bag_ = std::make_unique<util::BufferableBag>(path_to_bag_, 
        // NOTE: this REQUIRES OfflineSegmentationNode to be constructed with std::make_shared
        std::make_shared<util::BufferableBag::TFBridge>(node), 
        std::vector<std::string>{incoming_points_topic_});
    return bag_->message_count();
}

bool BagDataloader::finished() {
    return bag_ ? bag_->finished() : true;
}

std::pair<util::Cloud, std_msgs::msg::Header> BagDataloader::next() {
    // get next message from bag
    const rosbag2_storage::SerializedBagMessage serialized_bag_msg = bag_->PopNextMessage();
    const auto& topic = serialized_bag_msg.topic_name;
    const rclcpp::SerializedMessage msg_ser(*serialized_bag_msg.serialized_data);
    // this condition should always be truthy... but whatever...
    std::shared_ptr<sensor_msgs::msg::PointCloud2> msg = deserialize_next_msg<sensor_msgs::msg::PointCloud2>(msg_ser);
    return std::make_pair(msg, msg->header);
}
