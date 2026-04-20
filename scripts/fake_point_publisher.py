#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import struct
import random

rospy.init_node('fake_point_publisher')
pub = rospy.Publisher('/topic_points', PointCloud2, queue_size=10)
rate = rospy.Rate(1)  # 1 Hz

while not rospy.is_shutdown():
    points = []
    for i in range(0, random.randint(1, 10)):
        points.append((random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)))
    msg = PointCloud2()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = 'map'
    msg.height = 1
    msg.width = len(points)
    msg.is_bigendian = False
    msg.is_dense = True

    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]

    msg.point_step = 12 
    msg.row_step = msg.point_step * msg.width

    msg.data = b''.join([struct.pack('fff', *p) for p in points])

    pub.publish(msg)
    rate.sleep()