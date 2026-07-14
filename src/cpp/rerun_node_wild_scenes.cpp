#include "rerun_node.h"

// PARAMETERS
// path_to_scans [REQUIRED]
// path_to_result
// path_to_model
// frames

// TOPICS
// sub: incoming_points_topic [PointCloud2]

class RerunNodeWildScenes : public RerunNode {
public:
    explicit RerunNodeWildScenes() : RerunNode(false) {
        std::filesystem::path path_to_scans;
        util::param_req_path(this, "path_to_scans", path_to_scans, false, true);
        // get all file names for later
        for(const auto& entry : std::filesystem::directory_iterator(path_to_scans)) {
            if(entry.is_regular_file() && entry.path().extension() == ".bin") scans_.emplace_back(entry.path());
        }
        std::sort(scans_.begin(), scans_.end());
    }
protected:
    virtual size_t init() override {
        return scans_.size();
    }

    virtual bool finished() override {
        return index_ >= scans_.size();
    }

    virtual std::pair<util::Cloud, std_msgs::msg::Header> next() override {
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
        header.frame_id = RerunNodeWildScenes::LIDAR_FRAME;
        header.stamp.sec = static_cast<std_msgs::msg::Header::_stamp_type::_sec_type>(std::stoul(sec_str));
        header.stamp.nanosec = static_cast<std_msgs::msg::Header::_stamp_type::_nanosec_type>(std::stoul(nsec_str));
        // open the file and read the number of points
        std::ifstream f(path_to_scan, std::ios::binary);
        if(!f) throw std::runtime_error("failed to open: " + path_to_scan.string());
        f.seekg(0, std::ios::end);
        std::size_t bytes = f.tellg();
        f.seekg(0, std::ios::beg);
        if(bytes % (4 * sizeof(float)) != 0) throw std::runtime_error("malformed scan: " + path_to_scan.string());
        std::size_t point_count = bytes / (4 * sizeof(float));
        // allocate space for the cloud
        std::vector<Eigen::Vector3f> points;
        points.reserve(point_count);
        std::vector<float> intensity;
        points.reserve(point_count);
        // parse the cloud
        for(size_t i = 0; i < point_count; ++i) {
            // read point
            std::array<float, 3> xyz;
            f.read(reinterpret_cast<char*>(xyz.data()), 3 * sizeof(float));
            points[i].x() = xyz[0];
            points[i].y() = xyz[1];
            points[i].z() = xyz[2];
            // read 'remission'
            f.read(reinterpret_cast<char*>(&intensity[i]), sizeof(float));
        }
        // construct util::Cloud
        util::Cloud cloud{points};
        cloud.add_field("intensity", intensity);
        // return the frame data
        return std::make_pair(std::move(cloud), header);
    }
private:
    static constexpr std::string LIDAR_FRAME{"lidar_frame"};
    std::vector<std::filesystem::path> scans_;
    size_t index_{0};
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RerunNodeWildScenes>();
    node->run();
    rclcpp::shutdown();
    return 0;
}
