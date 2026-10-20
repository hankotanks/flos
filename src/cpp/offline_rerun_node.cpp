// rerun
#include <rerun.hpp>
// package
#include "odometry/ground_plane.h"
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
        static const std::string MEMORY_LIMIT = std::to_string(RerunNode::MEMORY_LIMIT) + "GB";
        if(visualize_) {
            rerun::SpawnOptions options;
            if(RerunNode::MEMORY_LIMIT > 0) options.memory_limit = MEMORY_LIMIT;
            rr_.spawn(options).throw_on_failure();
        }
    }
protected:
    virtual void publish_scan(TimeSystem::Stamp stamp, const util::Cloud& cloud) override {
        RerunNode::set_stamp(stamp);
        RerunNode::publish_points(cloud, "map/points");
    }

    virtual void publish_local_map(TimeSystem::Stamp stamp, const util::Cloud& cloud) override {
        RerunNode::set_stamp(stamp);
        RerunNode::publish_points(cloud, "map");
    }

    virtual void publish_local_map_belief(TimeSystem::Stamp stamp, const VoxelMap<3, SemanticCloud>::view_voxels& voxels, float voxel_size) override {
        RerunNode::set_stamp(stamp);
        std::map<uint8_t, std::vector<rerun::datatypes::Vec3D>> centers;
        std::map<uint8_t, std::vector<rerun::datatypes::Vec3D>> sizes;
        for(const auto& [voxel, belief, fill] : voxels) {
            if(fill < RerunNode::VOXEL_MAP_FILL_THRESHOLD) continue;
            centers[belief.prediction_].emplace_back(
                (static_cast<float>(voxel[0]) + 0.5f) * voxel_size,
                (static_cast<float>(voxel[1]) + 0.5f) * voxel_size,
                (static_cast<float>(voxel[2]) + 0.5f) * voxel_size);
            sizes[belief.prediction_].emplace_back(voxel_size, voxel_size, voxel_size);
        }
        for(const auto& [semantics, class_centers] : centers) {
            if(class_centers.empty()) continue;
            const auto& class_sizes = sizes.at(semantics);
            auto msg = rerun::archetypes::Boxes3D::from_centers_and_sizes(class_centers, class_sizes)
                .with_fill_mode(rerun::components::FillMode::Solid);;
            rr_.log("map/belief/" + std::to_string(semantics), msg);
        }
    }

    virtual void publish_pose(TimeSystem::Stamp stamp, const std_msgs::msg::Header::_frame_id_type& frame_id, const Sophus::SE3f& pose) override {
        RerunNode::set_stamp(stamp);
        const auto& t = pose.translation();
        rr_.log("trajectory", rerun::Points3D().with_positions({
            rerun::Position3D(t.x(), t.y(), t.z())
        }).with_radii(RerunNode::DEFAULT_POINT_RADIUS * 3.f));
    }

    virtual void publish_statistics(TimeSystem::Stamp stamp, const OdometrySystem::Statistics& statistics) override {
        RerunNode::set_stamp(stamp);
        rr_.log("statistics/iterations", rerun::Scalar(statistics.align.iterations));
        rr_.log("statistics/mean_correspondence_count", rerun::Scalar(statistics.align.mean_correspondence_count));
        rr_.log("statistics/velocity/linear", rerun::Scalar(statistics.velocity.linear));
        rr_.log("statistics/velocity/angular", rerun::Scalar(statistics.velocity.angular));
        float refined = statistics.align.refined ? 1.f : 0.f;
        rr_.log("statistics/refined", rerun::Scalar(refined));
        refined_mean_ = (static_cast<float>(count_) * refined_mean_ + refined) / static_cast<float>(count_ + 1.f);
        count_++;
        rr_.log("statistics/refined/mean", rerun::Scalar(refined_mean_));
    }

    virtual void publish_progress(TimeSystem::Stamp stamp, float percent, float remaining) override {
        RerunNode::set_stamp(stamp);
        rr_.log("progress/percent", rerun::Scalar(percent));
        rr_.log("progress/remaining", rerun::Scalar(remaining));
    }
    
    virtual void publish_ground(TimeSystem::Stamp stamp, const util::Cloud& cloud, const Eigen::Vector3f& centroid, const Eigen::Vector3f& normal) override {
        RerunNode::set_stamp(stamp);
        // plane
        const Eigen::Vector3f n = normal.normalized();
        // orthonormal vectors
        const Eigen::Vector3f ref = (std::abs(n.z()) < 0.9f) ? Eigen::Vector3f::UnitZ() : Eigen::Vector3f::UnitX();
        const Eigen::Vector3f t1 = n.cross(ref).normalized();
        const Eigen::Vector3f t2 = n.cross(t1).normalized();
        // vertices
        static constexpr size_t SEGMENTS{64}; // triangle fan 'resolution'
        std::vector<rerun::Position3D> vertices;
        vertices.reserve(SEGMENTS + 1);
        vertices.emplace_back(centroid.x(), centroid.y(), centroid.z());
        for(uint32_t i = 0; i < SEGMENTS; ++i) {
            const float theta = 2.f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(SEGMENTS);
            const Eigen::Vector3f p = centroid + odometry_config_.config.max_distance_ground * (std::cos(theta) * t1 + std::sin(theta) * t2);
            vertices.emplace_back(p.x(), p.y(), p.z());
        }
        // indices
        std::vector<rerun::components::TriangleIndices> triangles;
        triangles.reserve(SEGMENTS);
        for(uint32_t i = 0; i < SEGMENTS; ++i) {
            triangles.emplace_back(std::array<uint32_t, 3>{0, 1 + i, static_cast<uint32_t>((i + 1) % SEGMENTS) + 1});
        }
        rr_.log("map/ground/plane", rerun::archetypes::Mesh3D(vertices).with_triangle_indices(triangles));
        // points
        float inlier_distance = std::sqrt(odometry_config_.config.voxel_size * odometry_config_.config.voxel_size / static_cast<float>(odometry_config_.config.voxel_capacity));
        std::span<const uint8_t> predictions = cloud.field<uint8_t>("prediction");
        std::vector<rerun::Position3D> points;
        for(size_t i = 0; i < predictions.size(); ++i) {
            if(odometry_config_.config.projection.ground.count(predictions[i]) == 0) continue;
            if(normal.transpose() * (cloud[i] - centroid) > inlier_distance) continue;
            points.emplace_back(cloud[i].x(), cloud[i].y(), cloud[i].z());
        }
        rr_.log("map/ground/consensus", rerun::Points3D().with_positions(points));
    };

    virtual void publish_projection(TimeSystem::Stamp stamp, const VoxelMap<2, BirdsEyeCloud>& previous, const VoxelMap<2, BirdsEyeCloud>& current) override {
        RerunNode::set_stamp(stamp);
        // previous
        std::vector<rerun::Vec2D> points_previous;
        for(const auto& [point, data] : previous.points()) {
            points_previous.emplace_back(point.x(), point.y());
        }
        rr_.log("scan/projection/previous", rerun::Points2D(points_previous));
        // current
        std::vector<rerun::Vec2D> points_current;
        for(const auto& [point, data] : current.points()) {
            points_current.emplace_back(point.x(), point.y());
        }
        rr_.log("scan/projection/current", rerun::Points2D(points_current));
        
    }
private:
    void publish_points(const util::Cloud& cloud, const std::string& topic) {
        if(cloud.empty()) return;
        std::map<uint8_t, std::vector<rerun::Position3D>> scans;
        const auto prediction = cloud.field<uint8_t>("prediction");
        for(size_t i = 0; i < cloud.size(); ++i) {
            if(scans.count(prediction[i]) == 0) scans[prediction[i]] = {};
            const auto& point = cloud[i];
            scans[prediction[i]].emplace_back(point.x(), point.y(), point.z());
        }
        for(const auto& [semantics, points] : scans) {
            auto msg = rerun::Points3D().with_positions(points);
            rr_.log(topic + "/" + std::to_string(semantics), msg);
        }
    }

    void set_stamp(TimeSystem::Stamp stamp) {
        const int64_t ns = static_cast<int64_t>(stamp.sec) * 1'000'000'000LL + static_cast<int64_t>(stamp.nanosec);
        rr_.set_time_nanos("ros_ws", ns);
    }
private:
    // rerun
    static constexpr float DEFAULT_POINT_RADIUS{0.2f};
    static constexpr size_t MEMORY_LIMIT{4};
    static constexpr float VOXEL_MAP_FILL_THRESHOLD{0.5f};
    rerun::RecordingStream rr_;
private:
    size_t count_{0};
    float refined_mean_{0.f};
};


int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RerunNode>();
    node->run();
    rclcpp::shutdown();
    return 0;
}
