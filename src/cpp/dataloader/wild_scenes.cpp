#include "dataloader/wild_scenes.h"

// system
#include <fstream>
// util
#include "util/tinyply.h"

WildScenesDataloader::WildScenesDataloader(rclcpp::Node* node) : Dataloader(node) {
    bool valid = std::any_of(std::filesystem::directory_iterator(path_to_data_), std::filesystem::directory_iterator{},
        [](const auto& entry) { return entry.is_regular_file() && entry.path().extension() == ".ply"; });
    if(!valid) throw Dataloader::UnsupportedFormat{"no PLY files in data folder"};
}

size_t WildScenesDataloader::init(const std::shared_ptr<rclcpp::Node>& node) {
    // get all file names for later
    for(const auto& entry : std::filesystem::directory_iterator(path_to_data_)) {
        if(!entry.is_regular_file()) continue;
        std::filesystem::path path_to_scan = entry.path();
        if(entry.path().extension() != ".ply") continue; 
        // get timestamp
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
        // push the frame
        scans_.emplace_back(header, entry.path());
    }
    std::sort(scans_.begin(), scans_.end(), [](const auto& fst, const auto& snd) {
        if(fst.first.stamp.sec != snd.first.stamp.sec) return fst.first.stamp.sec < snd.first.stamp.sec;
        return fst.first.stamp.nanosec < snd.first.stamp.nanosec;
    });
    return scans_.size();
}

bool WildScenesDataloader::finished() {
    return index_ >= scans_.size();
}

std::pair<util::Cloud, std_msgs::msg::Header> WildScenesDataloader::next() {
    const auto& [header, path_to_scan] = scans_[index_++];
    // open the file and parse its metadata
    std::ifstream f(path_to_scan, std::ios::binary);
    tinyply::PlyFile f_ply;
    f_ply.parse_header(f);
    auto raw = f_ply.request_properties_from_element("vertex", { "x", "y", "z" });
    f_ply.read(f);
    const uint8_t* bytes = raw->buffer.get_const();
    const size_t stride = raw->buffer.size_bytes() / raw->count;
    // read the points
    std::vector<Eigen::Vector3f> points;
    points.reserve(raw->count);
    switch(raw->t) {
    case tinyply::Type::FLOAT32:
        for(size_t i = 0; i < raw->count; ++i) {
            const float* xyz = reinterpret_cast<const float*>(bytes + i * stride);
            points.emplace_back(xyz[0], xyz[1], xyz[2]);
        }
        break;
    case tinyply::Type::FLOAT64:
        for(size_t i = 0; i < raw->count; ++i) {
            const double* xyz = reinterpret_cast<const double*>(bytes + i * stride);
            points.emplace_back(static_cast<float>(xyz[0]), static_cast<float>(xyz[1]), static_cast<float>(xyz[2]));
        }
        break;
    default: throw std::runtime_error("points must be floating-point");
    }
    // construct util::Cloud
    util::Cloud cloud{points};
    // return the frame data
    return std::make_pair(std::move(cloud), header);
}
