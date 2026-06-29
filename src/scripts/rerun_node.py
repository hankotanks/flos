#!/usr/bin/env python3

# system
import threading
import collections

# numpy
import numpy as np

# ros imports
import rclpy, rclpy.node, rclpy.callback_groups, rclpy.qos, rclpy.time, sensor_msgs.msg, sensor_msgs_py.point_cloud2, tf2_msgs.msg

# rerun
import rerun as rr

def quat_to_mat(q):
    x, y, z, w = q

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype = np.float32)

class RerunPointCloudNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("rerun_node")

        rr.init("visualizer", spawn = True)

        self.subscribers: list[rclpy.Subscription] = []
        
        self.scan_buffer = collections.deque()
        self.scan_lock = threading.Lock()
        self.subscribe("/forest_segmentation/inference", sensor_msgs.msg.PointCloud2, self.scan_callback)

        self.tf_buffer = {}
        self.tf_lock = threading.Lock()
        self.subscribe("/tf", tf2_msgs.msg.TFMessage, self.tf_callback)

        self.create_timer(0.02, self.log)   # 50 Hz

    def subscribe(self, topic: str, msg_type: type, callback, latching: bool = False) -> None:
        """Adds a subscriber to a topic with the given message type and callback."""
        # `qos_profile` can either be an int (history depth) or a QoSProfile.
        # See: https://docs.ros.org/en/rolling/p/rclpy/rclpy.node.html#rclpy.node.Node.create_subscription
        qos_profile = rclpy.qos.QoSProfile(depth = 1, durability = rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL) if latching else 10
        sub = self.create_subscription(
            msg_type = msg_type,
            topic = topic,
            callback = callback,
            qos_profile = qos_profile,
            callback_group = rclpy.callback_groups.ReentrantCallbackGroup(),  # allow concurrent callbacks
        )
        self.subscribers.append(sub)

    def log(self):
        ready = []

        with self.scan_lock:
            remaining = collections.deque()

            while self.scan_buffer:
                scan = self.scan_buffer.popleft()

                with self.tf_lock:
                    transform = self.tf_buffer.pop(scan["stamp"], None)

                if transform is None:
                    remaining.append(scan)
                else:
                    ready.append((scan, transform))

            self.scan_buffer = remaining

        for scan, transform in ready:
            self.log_scan(scan, transform)

    def log_scan(self, scan, transform):
        points = scan["points"]
        semantics = scan["semantics"]

        t, q = transform
        points = (quat_to_mat(q) @ points.T).T + t

        rr.set_time_nanos("ros_time", scan["stamp"])
        for cls in np.unique(semantics):
            mask = semantics == cls
            rr.log(f"scan/class_{int(cls)}", rr.Points3D(
                positions = points[mask],
                radii = 0.02,
            ))

    def scan_callback(self, msg: sensor_msgs.msg.PointCloud2):
        field_names = [f.name for f in msg.fields]
        points = []
        semantics = []
        idx = field_names.index("class")
        for p in sensor_msgs_py.point_cloud2.read_points(msg, skip_nans = True, field_names = field_names):
            points.append([p[0], p[1], p[2]])
            assert "class" in field_names
            semantics.append(p[idx])
        if not points: return

        points = np.array(points, dtype = np.float32)
        semantics = np.array(semantics, dtype = np.int8)

        with self.scan_lock:
            self.scan_buffer.append({
                "frame": msg.header.frame_id,
                "stamp": rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds,
                "points": points,
                "semantics": semantics,
            })

    def tf_callback(self, msg: tf2_msgs.msg.TFMessage):
        for transform in msg.transforms:
            t = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ], dtype = np.float32)

            q = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ], dtype = np.float32)

            if transform.header.frame_id == 'odom_frame':
                with self.tf_lock:
                    self.tf_buffer[rclpy.time.Time.from_msg(transform.header.stamp).nanoseconds] = (t, q)

def main():
    rclpy.init()
    node = RerunPointCloudNode()
    try:
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()