#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import sys
from fusion_utils.radar_utils import pc2_numpy, dbscan_cluster, filter_zero
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose, Vector3, Point, Quaternion
import std_msgs.msg as std_msgs
from visualization_msgs.msg import MarkerArray, Marker
from rclpy.clock import Clock

def convert_bbox(boxes):
    Q = Quaternion(x=0., y=0., z=0., w=1.)
    # bboxes = []
    markers = []
    scale = Vector3(x=1., y=1., z=1.)
    header = std_msgs.Header(frame_id="map", stamp=Clock().now().to_msg())
    for i in range(boxes.shape[0]):
        # lwh = Vector3(x=boxes[i, :][3], y=boxes[i, :][4], z=boxes[i, :][5])
        xyz = Point(x=boxes[i, 0], y=boxes[i, 1], z=boxes[i, 2])

        P = Pose(position=xyz, orientation=Q)
        # jskbox = BoundingBox(header=header, pose=P, dimensions=lwh, value=np.float32(0.0),label=0)
        # bboxes.append(jskbox)

        marker = Marker(header=header, type=2, pose=P, scale=scale)
        marker.color.a = 1.0
        marker.color.r = 0.
        marker.color.g = 1.0
        marker.color.b = 0.
        markers.append(marker)
    return MarkerArray(markers=markers)


class Radar_Detection(Node):
    def __init__(self):
        super().__init__('Radar_sub')
        self.radar_subscriber = self.create_subscription(PointCloud2, '/Radar', self.listener_callback, 10)
        self.radar_publisher = self.create_publisher(MarkerArray, '/Detected_Radar_Objects', 10)

    def listener_callback(self, msg):
        npts = msg.width
        arr = pc2_numpy(msg, npts)
        # arr = filter_zero(arr)
        pc = arr[:, :4]
        total_box, cls = dbscan_cluster(pc, eps=2.5, min_sample=15)
        if cls:
            measSet = np.empty((0, 4))
            for ii, cc in enumerate(cls):
                centroid = np.mean(cc, axis=0)
                # get tracking measurement
                measSet = np.vstack((measSet, centroid))
            markers = convert_bbox(measSet)
            self.radar_publisher.publish(markers)



def radar_detection_node(args=None):
    rclpy.init(args=args)
    sub = Radar_Detection()
    rclpy.spin(sub)
    rclpy.shutdown()

if __name__ == '__main__':
    radar_detection_node()