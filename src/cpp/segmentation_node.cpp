// pybind
#include "forest_segmentor_bindings.h"
// system
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>
#include <optional>
#include <filesystem>

// eigen
#include <Eigen/Core>
// ros
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
// other
#include "Cloud.h"

class SegmentationNode : public rclcpp::Node {
public:
    SegmentationNode() : rclcpp::Node("forest_segmentation") {
        // get path to default model
        std::filesystem::path path_to_model_default{PROJECT_ROOT};
        while(path_to_model_default.filename() != PROJECT_NAME) 
            path_to_model_default = path_to_model_default.parent_path();
        path_to_model_default /= "models";
        path_to_model_default /= "default.ckpt";
        // declare topics and model path parameters
        this->declare_parameter<std::string>("incoming_points_topic", "/hesai/pandar");
        this->declare_parameter<std::string>("outgoing_points_topic", "/inference");
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
            std::bind(&SegmentationNode::callback, this, std::placeholders::_1));
        outgoing_points_publisher = this->create_publisher<sensor_msgs::msg::PointCloud2>(outgoing_points_topic, 10);
        // segmentation threads
        segmentation_thread = std::thread(&SegmentationNode::loop, this);
    }

    ~SegmentationNode() {
        buffer.stop();
        if(segmentation_thread.joinable()) segmentation_thread.join();
    }

private:
    struct Buffer {
        static constexpr size_t capacity = 5;

        void push(rclcpp::Logger logger, const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) noexcept {
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

        std::optional<std::pair<cloud::Cloud, std_msgs::msg::Header>> pop() {
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
        
        void stop() { 
            {
                std::lock_guard lock(mutex_);
                running_ = false; 
            }
            cv_.notify_all(); 
        }

    private:
        std::mutex mutex_;
        std::condition_variable cv_;
        std::queue<std::pair<cloud::Cloud, std_msgs::msg::Header>> buffer_;
        std::atomic<bool> running_{true};
    };

private:
    // parameters
    std::string incoming_points_topic;
    std::string outgoing_points_topic;
    std::string path_to_model;
    // topic stuff  
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr incoming_points_subscriber;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr outgoing_points_publisher;
    // buffer for incoming point data
    Buffer buffer;
    // separate thread for processing buffered clouds
    std::thread segmentation_thread;

private:
    void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
        buffer.push(this->get_logger(), msg);
    }

    void loop() {
        // hold the GIL
        py::scoped_interpreter guard{};
        // get the inference method from the python module
        py::object infer_module = py::module_::import("forest_segmentor.inference_helper");
        py::object infer_helper = infer_module.attr("InferenceHelper")(path_to_model);
        RCLCPP_INFO_ONCE(this->get_logger(), "segmentation_node successfully loaded inference model");
        py::object infer = infer_helper.attr("infer");
        // loop until node is stopped
        while(true) {
            // get the next batch of points
            auto maybe = buffer.pop();
            if(!maybe) break;
            auto [cloud, header] = std::move(*maybe);
            static_assert(std::is_same_v<decltype(cloud), cloud::Cloud>);
            static_assert(std::is_same_v<decltype(header), std_msgs::msg::Header>);
            // perform the inference
            const std::vector<Eigen::Vector3f>& points = cloud.get_points();
            // TODO: It would be GREAT if we could avoid constructing this vector every frame
            std::vector<float> intensity = cloud.get_field<float>("intensity");
            py::tuple result = infer(py::cast(points, py::return_value_policy::reference),
                py::cast(intensity, py::return_value_policy::reference)).cast<py::tuple>();
            // output
            cloud.add_field("class", result[0].cast<VectorUINT8>());
            cloud.add_field("confidence", result[1].cast<VectorFLOAT32>());
            outgoing_points_publisher->publish(std::move(cloud.message(header)));
        }
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SegmentationNode>());
    rclcpp::shutdown();
    return 0;
}
