#!/usr/bin/env python3

# numpy
import numpy as np

# ros imports
import rclpy
import rclpy.node, rclpy.callback_groups, rclpy.qos, rclpy.time
import sensor_msgs
import sensor_msgs.msg, sensor_msgs_py.point_cloud2
import nav_msgs
import nav_msgs.msg

# rerun
import rerun as rr

class RerunPointCloudNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("rerun_node")
        rr.init("visualizer", spawn = True)

        self.subscribers: list[rclpy.Subscription] = []
        # TODO: equivalent of PROJECT_NAME in python nodes
        self.subscribe("/forest_segmentation/map/points", sensor_msgs.msg.PointCloud2, self.scan_callback)
        self.subscribe("/forest_segmentation/odometry", nav_msgs.msg.Odometry, self.odom_callback)

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

    def scan_callback(self, msg: sensor_msgs.msg.PointCloud2):
        raw = np.array(list(sensor_msgs_py.point_cloud2.read_points(msg, skip_nans = True, field_names = ['x', 'y', 'z', 'class'])))
        points = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(np.float32)
        semantics = raw['class'].astype(np.uint8)

        rr.set_time_nanos("ros_time", rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds)
        for cls in np.unique(semantics):
            mask = semantics == cls
            rr.log(f"scan/{int(cls)}", rr.Points3D(
                positions = points[mask],
                radii = 0.02,
            ))

    def odom_callback(self, msg: nav_msgs.msg.Odometry) -> None:
        rr.set_time_nanos("ros_time", rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds)
        rr.log(f"pos", rr.Points3D(
            positions = [[msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]],
            radii = 0.2,
        ))

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