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
// other
#include "aliases.h"
#include "utils.h"

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
            const CloudWithIntensity &cloud = pointcloud2_with_intensity_to_eigen(msg);
            if(std::get<0>(cloud).empty()) return;
            { // ensure capacity isn't reached
                std::unique_lock lock(mutex_);
                if(buffer_.size() >= capacity) {
                    RCLCPP_WARN_ONCE(logger, "segmentation_node can't keep up!");
                    return;
                }
                // push the cloud
                buffer_.emplace(std::move(cloud), msg->header);
                cv_.notify_one();
            }   
        }

        std::optional<std::pair<CloudWithIntensity, std_msgs::msg::Header>> pop() {
            { // wait
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [this] { return !running_ || !buffer_.empty(); });
                if(!running_ && buffer_.empty()) return std::nullopt;
            }
            // pop points from buffer and return
            std::pair<CloudWithIntensity, std_msgs::msg::Header> cloud_and_header = std::move(buffer_.front());
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
        std::queue<std::pair<CloudWithIntensity, std_msgs::msg::Header>> buffer_;
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
            auto [cloud, header] = *maybe;
            static_assert(std::is_same_v<decltype(cloud), CloudWithIntensity>);
            static_assert(std::is_same_v<decltype(header), std_msgs::msg::Header>);
            auto [points, intensities] = cloud;
            static_assert(std::is_same_v<decltype(points), Cloud>);
            static_assert(std::is_same_v<decltype(intensities), FieldF32>);
            // perform the inference
            py::tuple result = infer(py::cast(points, py::return_value_policy::reference),
                py::cast(intensities, py::return_value_policy::reference)).cast<py::tuple>();
            // output
            std::map<std::string, Field> fields;
            fields.emplace(std::make_pair("intensity", intensities));
            fields.emplace(std::make_pair("class", result[0].cast<FieldU8>()));
            fields.emplace(std::make_pair("confidence", result[1].cast<FieldF32>()));
            outgoing_points_publisher->publish(pointcloud2_from_eigen(header, std::make_pair(points, fields)));
        }
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SegmentationNode>());
    rclcpp::shutdown();
    return 0;
}
