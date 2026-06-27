#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Callable, Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from tf2_msgs.msg import TFMessage
from rclpy.time import Time

import rerun as rr

class RerunPointCloudNode(Node):
    def __init__(self):
        super().__init__("rerun_node")
        rr.init("visualizer", spawn=True)
        self.subscribers: list[rclpy.Subscription] = []
        self.tf_buffer = {}
        self.subscribe("/forest_segmentation/inference", PointCloud2, self.scan_callback)
        self.subscribe("/tf", TFMessage, self.tf_callback)

    def subscribe(
        self, topic: str, msg_type: type, callback, latching: bool = False
    ) -> None:
        """Adds a subscriber to a topic with the given message type and callback."""
        # `qos_profile` can either be an int (history depth) or a QoSProfile.
        # See: https://docs.ros.org/en/rolling/p/rclpy/rclpy.node.html#rclpy.node.Node.create_subscription
        qos_profile = QoSProfile(depth=1, durability = QoSDurabilityPolicy.TRANSIENT_LOCAL) if latching else 10
        sub = self.create_subscription(
            msg_type=msg_type,
            topic=topic,
            callback=callback,
            qos_profile=qos_profile,
            callback_group=ReentrantCallbackGroup(),  # allow concurrent callbacks
        )
        self.subscribers.append(sub)

    def scan_callback(self, msg: PointCloud2):
        field_names = [f.name for f in msg.fields]
        points = []
        semantics = []
        idx = field_names.index("class")
        for p in pc2.read_points(msg, skip_nans = True, field_names = field_names):
            points.append([p[0], p[1], p[2]])
            assert "class" in field_names
            semantics.append(p[idx])
        if not points: return

        points = np.array(points, dtype = np.float32)
        semantics = np.array(semantics, dtype = np.int8)
        colors = self.semantic_to_color(semantics)

        if msg.header.frame_id in self.tf_buffer:
            t, q, _ = self.tf_buffer.pop(msg.header.frame_id)

            R = RerunPointCloudNode.quat_to_mat(q)

            points = (R @ points.T).T + t

        time = Time.from_msg(msg.header.stamp)
        rr.set_time_nanos("ros_time", time.nanoseconds)
        # TODO: log one cloud per semantic class
        rr.log("scan", rr.Points3D(positions = points, colors = colors, radii = 0.02))        

    def tf_callback(self, msg: TFMessage):
        for transform in msg.transforms:
            parent = transform.header.frame_id
            child = transform.child_frame_id

            t = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ], dtype=np.float32)

            q = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ], dtype=np.float32)

            self.tf_buffer[child] = (t, q, parent)

    def quat_to_mat(q):
        x, y, z, w = q

        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        return np.array([
            [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
            [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
        ], dtype=np.float32)

    def semantic_to_color(self, labels):
        rng = np.random.default_rng(42)
        unique = np.unique(labels)
        color_map = {}
        for u in unique:
            if u == -1:
                color_map[u] = [200, 200, 200]
            else:
                color_map[u] = rng.integers(0, 255, size=3)
        return np.array([color_map[l] for l in labels], dtype=np.uint8)

def main():
    rclpy.init()
    node = RerunPointCloudNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()