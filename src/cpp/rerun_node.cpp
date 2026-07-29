// system
#include <optional>
// rerun
#include <rerun.hpp>
// package
#include "offline_node.h"
// util
#include "util/cloud.h"

// PARAMETERS
// path_to_data [REQUIRED]
// path_to_result
// path_to_model
// frames
// visualize
// visualize_map
// incoming_points_topic [REQUIRED for BagDataloader]

// TOPICS
// sub: incoming_points_topic [PointCloud2]

class RerunNode : public OfflineNode {
public:
    explicit RerunNode() : OfflineNode(), rr_("visualizer") {
        if(!rclcpp::ok()) return;
        if(visualize_) {
            rerun::SpawnOptions options;
            if(RerunNode::MEMORY_LIMIT > 0) options.memory_limit = std::to_string(RerunNode::MEMORY_LIMIT) + "GB";
            rr_.spawn(options).throw_on_failure();
        }
    }
protected:
    virtual void publish_scan(const std_msgs::msg::Header::_stamp_type& stamp, const util::Cloud& cloud) override {
        RerunNode::set_stamp(stamp);
        RerunNode::publish_points(cloud, "scan", rerun::Color(255, 255, 255));
    }

    virtual void publish_local_map(const std_msgs::msg::Header::_stamp_type& stamp, const util::Cloud& cloud) override {
        RerunNode::set_stamp(stamp);
        RerunNode::publish_points(cloud, "map");
    }

    virtual void publish_pose(const std_msgs::msg::Header::_stamp_type& stamp, const std_msgs::msg::Header::_frame_id_type& frame_id, const Sophus::SE3f& pose) override {
        RerunNode::set_stamp(stamp);
        const auto& t = pose.translation();
        // TODO: I would prefer this be a LineStrips3D
        rr_.log("trajectory", rerun::Points3D().with_positions({
            rerun::Position3D(t.x(), t.y(), t.z())
        }).with_radii(RerunNode::DEFAULT_POINT_RADIUS * 3.f));
    }

    virtual void publish_statistics(const std_msgs::msg::Header::_stamp_type& stamp, const OdometrySystem::Statistics& statistics) override {
        RerunNode::set_stamp(stamp);
        rr_.log("statistics/iterations", rerun::Scalar(statistics.iterations));
        rr_.log("statistics/correspondences", rerun::Scalar(statistics.correspondences));
        rr_.log("statistics/velocity/linear", rerun::Scalar(statistics.velocity.linear));
        rr_.log("statistics/velocity/angular", rerun::Scalar(statistics.velocity.angular));
    }

    virtual void publish_progress(const std_msgs::msg::Header::_stamp_type& stamp, float percent, float remaining) override {
        RerunNode::set_stamp(stamp);
        rr_.log("progress/percent", rerun::Scalar(percent));
        rr_.log("progress/remaining", rerun::Scalar(remaining));
    }
private:
    void publish_points(const util::Cloud& cloud, const std::string& topic, std::optional<rerun::Color> single_color = std::nullopt, float radius_scalar = 1.f) {
        if(cloud.empty()) return;
        std::map<uint8_t, std::vector<rerun::Position3D>> scans;
        const auto semantics = cloud.field<uint8_t>("class");
        for(size_t i = 0; i < cloud.size(); ++i) {
            if(scans.count(semantics[i]) == 0) scans[semantics[i]] = {};
            const auto& point = cloud[i];
            scans[semantics[i]].emplace_back(point.x(), point.y(), point.z());
        }
        for(const auto& [semantics, points] : scans) {
            auto msg = rerun::Points3D()
                .with_positions(points)
                .with_radii(RerunNode::DEFAULT_POINT_RADIUS * radius_scalar);
            if(single_color) msg = std::move(msg).with_colors(*single_color);
            rr_.log(topic + "/" + std::to_string(semantics), msg);
        }
    }

    void set_stamp(const std_msgs::msg::Header::_stamp_type& stamp) {
        const int64_t ns = static_cast<int64_t>(stamp.sec) * 1'000'000'000LL + static_cast<int64_t>(stamp.nanosec);
        rr_.set_time_nanos("ros_ws", ns);
    }
private:
    static constexpr float DEFAULT_POINT_RADIUS{0.2f};
    static constexpr size_t MEMORY_LIMIT{4};
    rerun::RecordingStream rr_;
};


int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RerunNode>();
    node->run();
    rclcpp::shutdown();
    return 0;
}
