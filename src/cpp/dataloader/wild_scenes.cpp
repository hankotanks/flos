#include "dataloader/wild_scenes.h"

// system
#include <fstream>

WildScenesDataloader::WildScenesDataloader(rclcpp::Node* node) : Dataloader(node) {
    bool valid = std::any_of(std::filesystem::directory_iterator(path_to_data_), std::filesystem::directory_iterator{},
        [](const auto& entry) { return entry.is_regular_file() && entry.path().extension() == ".bin"; });
    if(!valid) throw Dataloader::UnsupportedFormat{"no BIN files in data folder"};
}

size_t WildScenesDataloader::init(const std::shared_ptr<rclcpp::Node>& node) {
    // get all file names for later
    for(const auto& entry : std::filesystem::directory_iterator(path_to_data_)) {
        if(entry.is_regular_file() && entry.path().extension() == ".bin") scans_.emplace_back(entry.path());
    }
    std::sort(scans_.begin(), scans_.end());
    return scans_.size();
}

bool WildScenesDataloader::finished() {
    return index_ >= scans_.size();
}

std::pair<util::Cloud, std_msgs::msg::Header> WildScenesDataloader::next() {
    const std::filesystem::path& path_to_scan = scans_[index_++];
    // get the timestamp
    std::string stamp_raw = path_to_scan.stem().string();
    std::replace(stamp_raw.begin(), stamp_raw.end(), '-', '.');
    const auto pos = stamp_raw.find('.');
    if(pos == std::string::npos) throw std::runtime_error("invalid timestamp: " + path_to_scan.string());
    const auto sec_str = stamp_raw.substr(0, pos);
    auto nsec_str = stamp_raw.substr(pos + 1);
    // see: https://github.com/csiro-robotics/WildScenes/blob/main/wildscenes/tools/utils.py:timestamp_to_bag_time
    nsec_str.append(9 - nsec_str.size(), '0');
    // construct header
    std_msgs::msg::Header header;
    header.frame_id = WildScenesDataloader::LIDAR_FRAME;
    header.stamp.sec = static_cast<std_msgs::msg::Header::_stamp_type::_sec_type>(std::stoul(sec_str));
    header.stamp.nanosec = static_cast<std_msgs::msg::Header::_stamp_type::_nanosec_type>(std::stoul(nsec_str));
    // open the file and read the number of points
    std::ifstream f(path_to_scan, std::ios::binary);
    if(!f) throw std::runtime_error("failed to open: " + path_to_scan.string());
    f.seekg(0, std::ios::end);
    std::size_t bytes = static_cast<std::size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    if(bytes % (3 * sizeof(float)) != 0) throw std::runtime_error("malformed scan: " + path_to_scan.string());
    std::size_t point_count = bytes / (3 * sizeof(float));
    // allocate space for the cloud
    std::vector<Eigen::Vector3f> points;
    points.reserve(point_count);
    std::vector<float> intensity;
    intensity.reserve(point_count);
    // parse the cloud
    for(size_t i = 0; i < point_count; ++i) {
        // read point
        std::array<float, 3> xyz;
        f.read(reinterpret_cast<char*>(xyz.data()), 3 * sizeof(float));
        points.emplace_back(xyz[0], xyz[1], xyz[2]);
        // TODO: these scans don't even contain intensity, we need to retrain the model to work without intensity
        intensity.emplace_back(1.f);
    }
    // construct util::Cloud
    util::Cloud cloud{points};
    cloud.add_field("intensity", intensity);
    // return the frame data
    return std::make_pair(std::move(cloud), header);
}
