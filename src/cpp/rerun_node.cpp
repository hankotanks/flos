// system
#include <optional>
// rerun
#include <rerun.hpp>
// package
#include "offline_node.h"
// util
#include "rerun/archetypes/line_strips3d.hpp"
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
        rr_.log("statistics/threshold/min", rerun::Scalar(statistics.threshold.min));
        rr_.log("statistics/threshold/max", rerun::Scalar(statistics.threshold.max));
        rr_.log("statistics/velocity/linear", rerun::Scalar(statistics.velocity.linear));
        rr_.log("statistics/velocity/angular", rerun::Scalar(statistics.velocity.angular));
    }

    virtual void publish_progress(const std_msgs::msg::Header::_stamp_type& stamp, float percent, float remaining) override {
        RerunNode::set_stamp(stamp);
        rr_.log("progress/percent", rerun::Scalar(percent));
        rr_.log("progress/remaining", rerun::Scalar(remaining));
    }

    virtual void publish_experimental(const std_msgs::msg::Header::_stamp_type& stamp, const Sophus::SE3f& pose, const OdometrySystem::Experimental& experimental) override {
        // ground plane in 3d
        float s = 10.f;
        Eigen::Vector3f normal = experimental.plane.col(0);
        Eigen::Vector3f t1 = experimental.plane.col(1);
        Eigen::Vector3f t2 = experimental.plane.col(2);
        Eigen::Vector3f p0 = (experimental.centroid + s * t1 + s * t2);
        Eigen::Vector3f p1 = (experimental.centroid + s * t1 - s * t2);
        Eigen::Vector3f p2 = (experimental.centroid - s * t1 - s * t2);
        Eigen::Vector3f p3 = (experimental.centroid - s * t1 + s * t2);
        RerunNode::set_stamp(stamp);
        rerun::LineStrip3D plane{{
            {p0.x(), p0.y(), p0.z()},
            {p1.x(), p1.y(), p1.z()},
            {p2.x(), p2.y(), p2.z()},
            {p3.x(), p3.y(), p3.z()},
            {p0.x(), p0.y(), p0.z()},
        }};
        rr_.log("scan/ground", rerun::LineStrips3D(plane));
        rr_.log("bev/accepted", rerun::Scalar(experimental.accepted ? 1.f : 0.f));
        // projected points
        if(experimental.accepted) {
            std::vector<rerun::Vec2D> points;
            for(const auto& p : experimental.bev_prev.points_projected) {
                points.emplace_back(p.x(), p.y());
            }
            rr_.log("scan/projection/prev", rerun::Points2D(points));
        } else {
            rr_.log("scan/projection/prev", rerun::Points2D());
        }
        if(experimental.accepted) {
            std::vector<rerun::Vec2D> points;
            for(const auto& p : experimental.bev_curr.points_projected) {
                points.emplace_back(p.x(), p.y());
            }
            rr_.log("scan/projection/curr", rerun::Points2D(points));
        } else {
            rr_.log("scan/projection/curr", rerun::Points2D());
        }
        if(experimental.accepted) {
            std::vector<rerun::Vec2D> points;
            for(const auto& p : experimental.bev_curr_post.points_projected) {
                points.emplace_back(p.x(), p.y());
            }
            rr_.log("scan/projection/curr_align", rerun::Points2D(points));
        } else {
            rr_.log("scan/projection/curr_align", rerun::Points2D());
        }
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
