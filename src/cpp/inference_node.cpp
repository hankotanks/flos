// pybind
#include "pybindings.h"
// system
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <queue>
#include <vector>
// ros
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

namespace {
    std::vector<Eigen::Vector3f> point_cloud2_to_eigen(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        const size_t point_count = static_cast<size_t>(msg->height) * msg->width;
        std::vector<Eigen::Vector3f> points;
        points.reserve(point_count);
        sensor_msgs::PointCloud2ConstIterator<float> msg_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> msg_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> msg_z(*msg, "z");
        for (size_t i = 0; i < point_count; ++i, ++msg_x, ++msg_y, ++msg_z) {
            points.emplace_back(*msg_x, *msg_y, *msg_z);
        }
        return points;
    }
}

class InferenceNode {
public:
    ros::NodeHandle nh;
    ros::Subscriber sub_points;
    std::atomic<bool> atomic_node_running = true;
    static constexpr std::chrono::seconds inference_publication_delay = std::chrono::seconds(1);
    std::mutex segmentation_mutex;
    std::thread segmentation_thread;
    std::atomic<bool> segmentation_thread_is_ready = false;

    std::condition_variable thread_ready_cv;
    std::queue<std::vector<Eigen::Vector3f>> segmentation_buffer;
    static constexpr size_t max_segmentation_buffer_size = 5;
    std::condition_variable segment_cv;
public:
    InferenceNode() : nh("~") {
        sub_points = nh.subscribe("/topic_points", 10, &InferenceNode::callback_points, this);

        segmentation_thread = std::thread([this]() { segmentation_loop(); });
        {
            std::unique_lock ready_lock(segmentation_mutex);
            thread_ready_cv.wait(ready_lock, [this]() { return segmentation_thread_is_ready.load(); });
        }
    }

    ~InferenceNode() {
        atomic_node_running = false;
        segment_cv.notify_all();
        thread_ready_cv.notify_all();
        segmentation_thread.join();
    }

    void callback_points(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        {
            std::lock_guard lock(segmentation_mutex);
            if(segmentation_buffer.size() >= max_segmentation_buffer_size) {
                ROS_WARN_STREAM_ONCE("Classification buffer limit being reached. Looks like the classification is unable to "
                                    "keep up with incoming lidar frames, probably "
                                    "consistently dropping lidar frames.");
                segment_cv.notify_one();
                return;
            }
        }
        const std::vector<Eigen::Vector3f> points = point_cloud2_to_eigen(msg);

        if(!points.empty()) {
            ROS_INFO_STREAM("Received " << points.size() << " points");
        }

        {
            std::lock_guard lock(segmentation_mutex);
            segmentation_buffer.emplace(points);
        }
        segment_cv.notify_one();
    }
    
    void segmentation_loop() {
        py::scoped_interpreter guard;
        py::object inference_helper = py::module_::import("forest_segmentor.inference_helper").attr("InferenceHelper")();
        segmentation_thread_is_ready = true;
        thread_ready_cv.notify_one();

        while(atomic_node_running) {
            std::unique_lock lock(segmentation_mutex);
            segment_cv.wait(lock, [this]() { return !atomic_node_running || !segmentation_buffer.empty(); });
            const std::vector<Eigen::Vector3f> points = segmentation_buffer.front();
            segmentation_buffer.pop();
            lock.unlock();

            py::object inference_result =
                inference_helper.attr("infer")(py::cast(points, py::return_value_policy::reference));
            std::vector<float> segmentation = inference_result.cast<std::vector<float>>();
            std::cout << "batch: { ";
            for(const float v : segmentation) {
                std::cout << v << ", ";
            }
            std::cout << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "forest_segmentation");
  InferenceNode node;
  ros::spin();
  return 0;
}
