#include "segmentation_node.h"

// tsl
#include <tsl/robin_map.h>
// ros
#include <rclcpp/logging.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>

template <>
struct std::hash<Eigen::Vector3i> {
    std::size_t operator()(const Eigen::Vector3i& voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
    }
};

namespace {
inline Eigen::Vector3i point_to_voxel(const Eigen::Vector3f& point, double voxel_size) {
    return {
        static_cast<int>(std::floor(point.x() / voxel_size)),
        static_cast<int>(std::floor(point.y() / voxel_size)),
        static_cast<int>(std::floor(point.z() / voxel_size))
    };
}

void voxel_downsample(util::Cloud &cloud, const double voxel_size) {
    tsl::robin_map<Eigen::Vector3i, size_t> grid;
    grid.reserve(cloud.size());
    const std::vector<Eigen::Vector3f>& points = cloud.points();
    for(size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3i voxel = point_to_voxel(points[i], voxel_size);
        if(!grid.contains(voxel)) grid.insert({voxel, i});
    }
    util::Cloud cloud_original{std::move(cloud)};
    std::for_each(grid.cbegin(), grid.cend(), [&](const auto& voxel_and_index) {
        cloud.add_point(cloud_original, voxel_and_index.second);
    });
}
} // private scope

SegmentationNode::SegmentationNode(const rclcpp::NodeOptions& options) : BufferNode(options, "incoming_points_topic", "/hesai/pandar"), voxel_size(std::nullopt) {
    // get path to default model
    std::filesystem::path path_to_model_default{PROJECT_ROOT};
    while(path_to_model_default.filename() != PROJECT_NAME) 
        path_to_model_default = path_to_model_default.parent_path();
    path_to_model_default /= "models";
    path_to_model_default /= "default.ckpt";
    // declare topics and model path parameters
    this->declare_parameter<std::string>("outgoing_points_topic", "/" PROJECT_NAME "/inference");
    this->declare_parameter<std::string>("path_to_model", path_to_model_default.string());
    rcl_interfaces::msg::ParameterDescriptor voxel_size_desc{};
    voxel_size_desc.dynamic_typing = true;
    this->declare_parameter("voxel_size", rclcpp::ParameterValue{}, voxel_size_desc);
    // store the parameters
    this->get_parameter("outgoing_points_topic", outgoing_points_topic);
    this->get_parameter("path_to_model", path_to_model);
    double voxel_size_temp; // NOTE: extra careful because I'm not sure of get_parameter's type inferencing behavior
    if(this->get_parameter("voxel_size", voxel_size_temp)) voxel_size = voxel_size_temp;
    // ensure that model path exists
    if(path_to_model.empty()) path_to_model = path_to_model_default;
    if(!std::filesystem::exists(path_to_model))
        throw std::runtime_error("model does not exist: " + path_to_model);
    // point publisher (subscriber is now handled by BufferNode)
    outgoing_points_publisher = this->create_publisher<sensor_msgs::msg::PointCloud2>(outgoing_points_topic, 10);
    // segmentation thread
    segmentation_thread = std::thread(&SegmentationNode::loop, this);
}

void SegmentationNode::loop() {
    // hold the GIL
    py::scoped_interpreter guard{};
    // get the inference method from the python module
    py::object infer_module = py::module_::import("forest_segmentor.inference_helper");
    py::object infer_helper = infer_module.attr("InferenceHelper")(path_to_model);
    RCLCPP_INFO_ONCE(this->get_logger(), "segmentation_node successfully loaded inference model");
    py::object infer = infer_helper.attr("infer");
    // loop until node is stopped
    while(rclcpp::ok()) {
        // get the next batch of points
        auto maybe = BufferNode::pop();
        if(!maybe) break;
        active = true;
        auto [cloud, header] = std::move(*maybe);
        static_assert(std::is_same_v<decltype(cloud), util::Cloud>);
        static_assert(std::is_same_v<decltype(header), std_msgs::msg::Header>);
        // perform the inference
        this->infer(infer, cloud, header);
        // publish the result
        outgoing_points_publisher->publish(std::move(cloud.message(header)));
        active = false;
    }
}

void SegmentationNode::infer(py::object infer, util::Cloud& cloud, const std_msgs::msg::Header& header) noexcept {
    py::object points = cloud.points_as_py_object();
    py::object intensity = cloud.field_as_py_object<float>("intensity");
    // perform inference
    py::tuple result = infer(points, intensity).cast<py::tuple>();
    // output
    cloud.add_field_from_py_object<uint8_t>("class", result[0]);
    cloud.add_field_from_py_object<float>("confidence", result[1]);
}

std::pair<util::Cloud, std_msgs::msg::Header> SegmentationNode::read(const std::shared_ptr<sensor_msgs::msg::PointCloud2>& msg) const noexcept {
    util::Cloud cloud{msg};
    if(voxel_size) voxel_downsample(cloud, *voxel_size);
    return std::make_pair(std::move(cloud), msg->header);
}

#if 0
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SegmentationNode>();
    rclcpp::spin(std::static_pointer_cast<rclcpp::Node>(node));
    rclcpp::shutdown();
    return 0;
}
#endif

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(SegmentationNode)