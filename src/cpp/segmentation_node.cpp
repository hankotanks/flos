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
        incoming_points_subscriber = handle.subscribe("/topic_points", 10, &SegmentationNode::callback, this);
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
            auto points = point_cloud2_to_eigen(msg);
            if(points.empty()) return;
            { // ensure capacity isn't reached
                std::unique_lock lock(mutex_);
                if(buffer_.size() >= capacity) {
                    ROS_WARN_STREAM_ONCE("segmentation_node can't keep up!");
                    return;
                }
            }
            // push the cloud
            buffer_.push(std::move(points));
            cv_.notify_one();
        }

        std::optional<std::vector<Eigen::Vector3f>> pop() {
            { // wait
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [this] { return !running_ || !buffer_.empty(); });
                if(!running_ && buffer_.empty()) return std::nullopt;
            }
            // pop points from buffer and return
            std::vector<Eigen::Vector3f> points = std::move(buffer_.front());
            buffer_.pop();
            return points;
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
        std::queue<std::vector<Eigen::Vector3f>> buffer_;
        std::atomic<bool> running_{true};
    };

private:
    ros::NodeHandle handle;
    ros::Subscriber incoming_points_subscriber;
    // buffer for incoming point data
    Buffer buffer;
    // separate thread for processing buffered clouds
    std::thread segmentation_thread;

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
            std::vector<Eigen::Vector3f> points = *maybe;
            // perform the inference
            py::object result = infer(py::cast(points, py::return_value_policy::reference));
            std::vector<float> classes = result.cast<std::vector<float>>();
            // output
            std::cout << "batch: { ";
            for(float v : classes) std::cout << v << ", ";
            std::cout << "}" << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "forest_segmentation");
  SegmentationNode node;
  ros::spin();
  return 0;
}
