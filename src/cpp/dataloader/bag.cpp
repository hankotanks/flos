#include "dataloader/bag.h"

// util
#include "util/rosargs.h"

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
    util::param_req(node, "incoming_points_topic", incoming_points_topic_);
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
