#include "base_node.h"

namespace util {
BaseNode::BaseNode() : rclcpp::Node(PROJECT_NAME) {
    // get path to default model
    std::filesystem::path path_to_model_default{PROJECT_ROOT};
    while(path_to_model_default.filename() != PROJECT_NAME) 
        path_to_model_default = path_to_model_default.parent_path();
    path_to_model_default /= "models";
    path_to_model_default /= "default.ckpt";
    // declare topics and model path parameters
    this->declare_parameter<std::string>("incoming_points_topic", "/hesai/pandar");
    this->declare_parameter<std::string>("outgoing_points_topic", "/" PROJECT_NAME "/inference");
    this->declare_parameter<std::string>("path_to_model", path_to_model_default.string());
    // store the parameters
    this->get_parameter("incoming_points_topic", incoming_points_topic);
    this->get_parameter("outgoing_points_topic", outgoing_points_topic);
    this->get_parameter("path_to_model", path_to_model);
    // ensure that model path exists
    if(path_to_model.empty()) path_to_model = path_to_model_default;
    if(!std::filesystem::exists(path_to_model))
        throw std::runtime_error("model does not exist: " + path_to_model);
    // point publisher and subscriber
    incoming_points_subscriber = this->create_subscription<sensor_msgs::msg::PointCloud2>(incoming_points_topic, 10,
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { this->callback(msg); });
    outgoing_points_publisher = this->create_publisher<sensor_msgs::msg::PointCloud2>(outgoing_points_topic, 10);
    // segmentation threads
    segmentation_thread = std::thread(&BaseNode::loop, this);
}

void BaseNode::Buffer::push(rclcpp::Logger logger, const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) noexcept {
    if(msg->width * msg->height == 0) return;
    { // ensure capacity isn't reached
        std::unique_lock lock(mutex_);
        if(buffer_.size() >= capacity) {
            RCLCPP_WARN_ONCE(logger, "segmentation_node can't keep up!");
            return;
        }
        // push the cloud
        buffer_.emplace(msg, msg->header);
        cv_.notify_one();
    }   
}

std::optional<std::pair<util::Cloud, std_msgs::msg::Header>> BaseNode::Buffer::pop() noexcept {
    { // wait
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return !running_ || !buffer_.empty(); });
        if(!running_ && buffer_.empty()) return std::nullopt;
    }
    // pop points from buffer and return
    auto cloud_and_header = std::move(buffer_.front());
    buffer_.pop();
    return cloud_and_header;
}

void BaseNode::Buffer::stop() noexcept { 
    {
        std::lock_guard lock(mutex_);
        running_ = false; 
    }
    cv_.notify_all(); 
}

void BaseNode::Buffer::wait_until_empty(const std::atomic<bool>& segmentation_active) noexcept {
    while(rclcpp::ok()) {
        { // wait for buffer to drain
            std::lock_guard<std::mutex> lock(mutex_);
            if(buffer_.empty() && !segmentation_active.load()) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
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
        auto maybe = buffer.pop();
        if(!maybe) break;
        segmentation_active = true;
        auto [cloud, header] = std::move(*maybe);
        static_assert(std::is_same_v<decltype(cloud), util::Cloud>);
        static_assert(std::is_same_v<decltype(header), std_msgs::msg::Header>);
        // perform the inference
        py::object points = cloud.points_as_py_object();
        py::object intensity = cloud.field_as_py_object<float>("intensity");
        py::tuple result = infer(points, intensity).cast<py::tuple>();
        // output
        cloud.add_field_from_py_object<uint8_t>("class", result[0]);
        cloud.add_field_from_py_object<float>("confidence", result[1]);
        outgoing_points_publisher->publish(std::move(cloud.message(header)));
        segmentation_active = false;
    }
    buffer.stop();
}
}