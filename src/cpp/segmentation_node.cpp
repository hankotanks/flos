// ros
#include <rclcpp/logging.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
// package
#include "segmentation/system.h"
// util
#include "util/rosargs.h"
#include "util/buffer_node.h"
#include "util/cloud.h"
#include "util/profiler.h"

// PARAMETERS
// debug
// incoming_points_topic
// path_to_model
// voxel_size
// max_distance
// max_distance_neighborhood
// ignore_ground
// ignore_trunks
// ignore_canopy
// ignore_shrubs
// nearby_ground
// nearby_trunks
// nearby_canopy
// nearby_shrubs

// TOPICS
// sub: incoming_points_topic (default: $/raw/points) [PointCloud2]
// pub: $/inference [PointCloud2]

class SegmentationNode : 
    util::Profiler, 
    public BufferNode<std::pair<util::Cloud, std_msgs::msg::Header>, sensor_msgs::msg::PointCloud2> {
public:
    explicit SegmentationNode() : util::Profiler(NODE_NAME),
        BufferNode("incoming_points_topic", "/" PROJECT_NAME "/raw/points"),
        config_(this),
        system_(config_.config) {
        // get path to default model
        std::filesystem::path path_to_model_default{PROJECT_ROOT};
        while(path_to_model_default.filename() != PROJECT_NAME) 
            path_to_model_default = path_to_model_default.parent_path();
        path_to_model_default /= "models";
        path_to_model_default /= "default.ckpt";
        // declare topics and model path parameters
        this->declare_parameter<std::string>("path_to_model", path_to_model_default.string());
        // store the parameters
        this->get_parameter("path_to_model", path_to_model_);
        // ensure that model path exists
        if(path_to_model_.empty()) path_to_model_ = path_to_model_default;
        if(!std::filesystem::exists(path_to_model_))
            throw std::runtime_error("model does not exist: " + path_to_model_);
        // point publisher (subscriber is now handled by BufferNode)
        outgoing_points_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/" PROJECT_NAME "/inference", 10);
        // start the worker
        BufferNode::start();
    }
public:
    SegmentationNode(const SegmentationNode&) = delete;
    SegmentationNode(SegmentationNode&&) = delete;
    SegmentationNode& operator=(const SegmentationNode&) = delete;
    SegmentationNode& operator=(SegmentationNode&&) = delete;
    ~SegmentationNode() { 
        BufferNode::stop(); 
    }
protected:
    struct Config {
        SegmentationSystem::Config config;
        Config(rclcpp::Node* node);
    };
protected:
    // config
    Config config_;
    // system
    SegmentationSystem system_;
    // parameters
    std::string path_to_model_;
    // topic stuff  
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr outgoing_points_publisher_;
protected:
    virtual void loop() noexcept override final {
        py::scoped_interpreter guard{};
        // initialize the system
        SegmentationSystem::Model model = system_.load(path_to_model_);
        RCLCPP_INFO_ONCE(this->get_logger(), NODE_NAME " successfully loaded inference model");
        // loop until node is stopped
        while(rclcpp::ok()) {
            // get the next batch of points
            auto maybe = BufferNode::pop();
            if(!maybe) break;
            auto [cloud, header] = std::move(*maybe);
            static_assert(std::is_same_v<decltype(cloud), util::Cloud>);
            static_assert(std::is_same_v<decltype(header), std_msgs::msg::Header>);
            { // perform the inference
                util::Profiler profiler("inference", this);
                model.infer(cloud, true);
            }
            // publish the result
            outgoing_points_publisher_->publish(std::move(cloud.message(header)));
        }
    }

    // overload of the BufferNode::read method
    virtual std::pair<util::Cloud, std_msgs::msg::Header> read(const std::shared_ptr<sensor_msgs::msg::PointCloud2>& msg) noexcept final override {
        util::Cloud cloud{msg};
        system_.preprocess(cloud);
        return std::make_pair(std::move(cloud), msg->header);
    }
};

// TODO: this node specifically causes segfaults on SIGINT
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SegmentationNode>();
    rclcpp::spin(std::static_pointer_cast<rclcpp::Node>(node));
    rclcpp::shutdown();
    return 0;
}

SegmentationNode::Config::Config(rclcpp::Node* node) {
    util::opt_param<float>(node, config.voxel_size, "voxel_size");
    util::opt_param<float>(node, config.max_distance, "max_distance");
    util::opt_param<float>(node, config.max_distance_neighborhood, "max_distance_neighborhood");
    // TODO: I would like my system to be class-name agnostic, so I've moved these parameters here
    // however, the actual class numbers should not be hard-coded...
    // this is a problem for a different day
    bool ignore_ground = config.ignore.count(1) > 0;
    bool ignore_trunks = config.ignore.count(3) > 0;
    bool ignore_canopy = config.ignore.count(4) > 0;
    bool ignore_shrubs = config.ignore.count(2) > 0;
    util::opt_param<bool>(node, ignore_ground, "ignore_ground");
    util::opt_param<bool>(node, ignore_trunks, "ignore_trunks");
    util::opt_param<bool>(node, ignore_canopy, "ignore_canopy");
    util::opt_param<bool>(node, ignore_shrubs, "ignore_shrubs");
    ignore_ground ? (config.ignore.insert(1), 0) : config.ignore.erase(1);
    ignore_trunks ? (config.ignore.insert(3), 0) : config.ignore.erase(3);
    ignore_canopy ? (config.ignore.insert(4), 0) : config.ignore.erase(4);
    ignore_shrubs ? (config.ignore.insert(2), 0) : config.ignore.erase(2);
    // TODO: similar situation with SegmentationSystem::Config::nearby
    bool nearby_ground = config.nearby.count(1) > 0;
    bool nearby_trunks = config.nearby.count(3) > 0;
    bool nearby_canopy = config.nearby.count(4) > 0;
    bool nearby_shrubs = config.nearby.count(2) > 0;
    util::opt_param<bool>(node, nearby_ground, "nearby_ground");
    util::opt_param<bool>(node, nearby_trunks, "nearby_trunks");
    util::opt_param<bool>(node, nearby_canopy, "nearby_canopy");
    util::opt_param<bool>(node, nearby_shrubs, "nearby_shrubs");
    nearby_ground ? (config.nearby.insert(1), 0) : config.nearby.erase(1);
    nearby_trunks ? (config.nearby.insert(3), 0) : config.nearby.erase(3);
    nearby_canopy ? (config.nearby.insert(4), 0) : config.nearby.erase(4);
    nearby_shrubs ? (config.nearby.insert(2), 0) : config.nearby.erase(2);
    // log parameters
    RCLCPP_INFO_STREAM(node->get_logger(), 
        std::boolalpha << "SegmentationSystem::Config" << std::endl <<
            "\tvoxel_size: " << config.voxel_size << "," << std::endl << 
            "\tmax_distance: " << config.max_distance << "," << std::endl << 
            "\tmax_distance_neighborhood: " << config.max_distance_neighborhood << "," << std::endl << 
            "\tignore_ground: " << ignore_ground << "," << std::endl << 
            "\tignore_trunks: " << ignore_trunks << "," << std::endl << 
            "\tignore_canopy: " << ignore_canopy << "," << std::endl << 
            "\tignore_shrubs: " << ignore_shrubs << "," << std::endl << 
            "\tnearby_ground: " << nearby_ground << "," << std::endl << 
            "\tnearby_trunks: " << nearby_trunks << "," << std::endl << 
            "\tnearby_canopy: " << nearby_canopy << "," << std::endl << 
            "\tnearby_shrubs: " << nearby_shrubs);
}
