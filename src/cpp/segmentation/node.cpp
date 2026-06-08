#include "node.h"

// tsl
#include <pybind11/embed.h>
#include <tsl/robin_map.h>
// ros
#include <rclcpp/logging.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
// util
#include "util/voxel.h"
#include "util/rosargs.h"

SegmentationNode::SegmentationNode() : 
    BufferNode("incoming_points_topic", "/hesai/pandar"),
    config_(SegmentationNode::configure(this)) {
    // get path to default model
    std::filesystem::path path_to_model_default{PROJECT_ROOT};
    while(path_to_model_default.filename() != PROJECT_NAME) 
        path_to_model_default = path_to_model_default.parent_path();
    path_to_model_default /= "models";
    path_to_model_default /= "default.ckpt";
    // declare topics and model path parameters
    this->declare_parameter<std::string>("outgoing_points_topic", "/" PROJECT_NAME "/inference");
    this->declare_parameter<std::string>("path_to_model", path_to_model_default.string());
    // store the parameters
    this->get_parameter("outgoing_points_topic", outgoing_points_topic_);
    this->get_parameter("path_to_model", path_to_model_);
    // ensure that model path exists
    if(path_to_model_.empty()) path_to_model_ = path_to_model_default;
    if(!std::filesystem::exists(path_to_model_))
        throw std::runtime_error("model does not exist: " + path_to_model_);
    // point publisher (subscriber is now handled by BufferNode)
    outgoing_points_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(outgoing_points_topic_, 10);
    // segmentation thread
    segmentation_thread_ = std::thread(&SegmentationNode::loop, this);
}

void SegmentationNode::loop() {
    // hold the GIL
    py::scoped_interpreter guard{};
    // get the inference method from the python module
    py::object infer_module = py::module_::import("forest_segmentor.inference_helper");
    py::object infer_helper = infer_module.attr("InferenceHelper")(path_to_model_);
    RCLCPP_INFO_ONCE(this->get_logger(), NODE_NAME " successfully loaded inference model");
    py::object infer = infer_helper.attr("infer");
    // loop until node is stopped
    while(rclcpp::ok()) {
        // get the next batch of points
        auto maybe = BufferNode::pop();
        if(!maybe) break;
        active_ = true;
        auto [cloud, header] = std::move(*maybe);
        static_assert(std::is_same_v<decltype(cloud), util::Cloud>);
        static_assert(std::is_same_v<decltype(header), std_msgs::msg::Header>);
        // perform the inference
        this->infer(infer, cloud);
        // publish the result
        outgoing_points_publisher_->publish(std::move(cloud.message(header)));
        active_ = false;
    }
}

void SegmentationNode::infer(py::object infer, util::Cloud& cloud) {
    py::object points = cloud.points_as_py_object();
    py::object intensity = cloud.field_as_py_object<float>("intensity");
    // perform inference
    py::object result = infer(points, intensity).cast<py::tuple>();
    if(!py::isinstance<py::tuple>(result)) throw std::runtime_error("inference failed");
    py::tuple result_fields = result.cast<py::tuple>();
    // output
    cloud.add_field_from_py_object<uint8_t>("class", result_fields[0]);
    cloud.add_field_from_py_object<float>("confidence", result_fields[1]);
    cloud.filter<uint8_t>("class", [this](const uint8_t& scalar) {
        return (scalar != 0) && !(this->config_.ignore_shrubs && scalar == 2) && !(this->config_.ignore_canopy && scalar == 4);
    });
}

std::pair<util::Cloud, std_msgs::msg::Header> SegmentationNode::read(const std::shared_ptr<sensor_msgs::msg::PointCloud2>& msg) noexcept {
    util::Cloud cloud{msg};
    util::voxel_downsample(cloud, config_.voxel_size * SegmentationNode::VOXEL_SIZE_SCALAR);
    cloud.filter_by_distance(Eigen::Vector3f::Zero(), config_.max_distance * SegmentationNode::MAX_DISTANCE_SCALAR);
    return std::make_pair(std::move(cloud), msg->header);
}

// TODO: Consider the upper bounds of these parameters
SegmentationNode::Config SegmentationNode::configure(rclcpp::Node* node) {
    SegmentationNode::Config config;
    auto ignore_canopy = util::numeric_param<bool>(node, "ignore_canopy");
    if(ignore_canopy) config.ignore_canopy = *ignore_canopy;
    auto ignore_shrubs = util::numeric_param<bool>(node, "ignore_shrubs");
    if(ignore_shrubs) config.ignore_shrubs = *ignore_shrubs;
    auto voxel_size = util::numeric_param<double>(node, "voxel_size");
    if(voxel_size) config.voxel_size = *voxel_size;
    auto max_distance = util::numeric_param<double>(node, "max_distance");
    if(max_distance) config.max_distance = *max_distance;
    return config;
}