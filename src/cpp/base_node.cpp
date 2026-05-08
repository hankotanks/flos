#include "base_node.h"
#include <rclcpp/logging.hpp>

BaseNode::BaseNode() {
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
    this->get_parameter("outgoing_points_topic", outgoing_points_topic);
    this->get_parameter("path_to_model", path_to_model);
    // ensure that model path exists
    if(path_to_model.empty()) path_to_model = path_to_model_default;
    if(!std::filesystem::exists(path_to_model))
        throw std::runtime_error("model does not exist: " + path_to_model);
    // point publisher (subscriber is now handled by BufferNode)
    outgoing_points_publisher = this->create_publisher<sensor_msgs::msg::PointCloud2>(outgoing_points_topic, 10);
    // segmentation thread
    segmentation_thread = std::thread(&BaseNode::loop, this);
}

void BaseNode::loop() {
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

void BaseNode::infer(py::object infer, util::Cloud& cloud, const std_msgs::msg::Header& header) noexcept {
    py::object points = cloud.points_as_py_object();
    py::object intensity = cloud.field_as_py_object<float>("intensity");
    // perform inference
    py::tuple result = infer(points, intensity).cast<py::tuple>();
    // output
    cloud.add_field_from_py_object<uint8_t>("class", result[0]);
    cloud.add_field_from_py_object<float>("confidence", result[1]);
}
