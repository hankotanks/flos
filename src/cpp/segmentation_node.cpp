// pybind
#include "forest_segmentor_bindings.h"
// system
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>
#include <vector>
#include <optional>
// eigen
#include <Eigen/Core>
// ros
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
// other
#include "utils.h"

class SegmentationNode {
public:
    SegmentationNode() : handle("~") {
        incoming_points_subscriber = handle.subscribe("/hesai/pandar", 10, &SegmentationNode::callback, this);
        segmentation_publisher = handle.advertise<sensor_msgs::PointCloud2>("/segmented_points", 10);
        segmentation_thread = std::thread(&SegmentationNode::loop, this);
    }

    ~SegmentationNode() {
        buffer.stop();
        if(segmentation_thread.joinable()) segmentation_thread.join();
    }

private:
    struct Buffer {
        static constexpr size_t capacity = 5;

        void push(const sensor_msgs::PointCloud2::ConstPtr& msg) noexcept {
            const auto& points_and_intensities = point_cloud2_to_eigen_with_intensities(msg);
            if(std::get<0>(points_and_intensities).empty()) return;
            { // ensure capacity isn't reached
                std::unique_lock lock(mutex_);
                if(buffer_.size() >= capacity) {
                    ROS_WARN_STREAM_ONCE("segmentation_node can't keep up!");
                    return;
                }
            }
            // push the cloud
            buffer_.push(std::move(points_and_intensities));
            cv_.notify_one();
        }

        std::optional<std::pair<std::vector<Eigen::Matrix<float, 3, 1>>, std::vector<float>>> pop() {
            { // wait
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [this] { return !running_ || !buffer_.empty(); });
                if(!running_ && buffer_.empty()) return std::nullopt;
            }
            // pop points from buffer and return
            auto points_and_intensities = std::move(buffer_.front());
            buffer_.pop();
            return points_and_intensities;
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
        std::queue<std::pair<std::vector<Eigen::Matrix<float, 3, 1>>, std::vector<float>>> buffer_;
        std::atomic<bool> running_{true};
    };

private:
    ros::NodeHandle handle;
    ros::Subscriber incoming_points_subscriber;
    ros::Publisher segmentation_publisher;
    // buffer for incoming point data
    Buffer buffer;
    // separate thread for processing buffered clouds
    std::thread segmentation_thread;
    // id of outgoing point cloud messages
    unsigned int seq = 0;

private:
    void callback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        buffer.push(msg);
    }

    void loop() {
        // hold the GIL
        py::scoped_interpreter guard{};
        // get the inference method from the python module
        py::object infer = py::module_::import("forest_segmentor.inference_helper").attr("InferenceHelper")().attr("infer");

        while(true) {
            // get the next batch of points
            auto maybe = buffer.pop();
            if(!maybe) break;
            auto [points, intensities] = *maybe;
            // perform the inference
            py::object result = infer(py::cast(points, py::return_value_policy::reference),
                py::cast(intensities, py::return_value_policy::reference));
            // output
            std::map<std::string, std::vector<float>> fields;
            fields.emplace(std::make_pair("intensity", intensities));
            fields.emplace(std::make_pair("classes", result.cast<std::vector<float>>()));
            std_msgs::Header header;
            header.seq = seq++;
            header.stamp = ros::Time::now();
            header.frame_id = "base_link";
            segmentation_publisher.publish(eigen_to_point_cloud2_with_fields(points, fields, header));
        }
    }
};

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "forest_segmentation");
  SegmentationNode node;
  ros::spin();
  return 0;
}
