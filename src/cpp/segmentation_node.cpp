// pybind
#include "forest_segmentor_bindings.h"
// system
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>
#include <vector>
// eigen
#include <Eigen/Core>
// ros
#include <ros/console.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
// other
#include "utils.h"

class SegmentationNode {
public:
    ros::NodeHandle handle;
    ros::Subscriber incoming_points_subscriber;

    std::thread segmentation_thread;
    std::atomic<bool> running{true};

    std::mutex buffer_mutex;
    std::condition_variable buffer_cv;
    std::queue<std::vector<Eigen::Vector3f>> buffer;
    static constexpr size_t max_buffer_size = 5;

    SegmentationNode() : handle("~") {
        incoming_points_subscriber = handle.subscribe("/topic_points", 10, &SegmentationNode::callback, this);
        segmentation_thread = std::thread(&SegmentationNode::loop, this);
    }

    ~SegmentationNode() {
        running = false;
        buffer_cv.notify_all();
        if(segmentation_thread.joinable()) segmentation_thread.join();
    }

    void callback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        auto points = point_cloud2_to_eigen(msg);
        if(points.empty()) return;

        std::unique_lock lock(buffer_mutex);
        if(buffer.size() >= max_buffer_size) {
            ROS_WARN_STREAM_ONCE("segmentation_node can't keep up!");
            return;
        }

        buffer.push(std::move(points));
        buffer_cv.notify_one();
    }

    void loop() {
        // hold the GIL lock
        py::scoped_interpreter guard{};
        // get the inference method from the python module
        py::object infer = py::module_::import("forest_segmentor.inference_helper").attr("InferenceHelper")().attr("infer");

        while(running) {
            std::vector<Eigen::Vector3f> points;
            {
                std::unique_lock lock(buffer_mutex);
                buffer_cv.wait(lock, [this] { return !running || !buffer.empty(); });
                if(!running && buffer.empty()) break;
                // pop points from buffer
                points = std::move(buffer.front());
                buffer.pop();
            }
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
