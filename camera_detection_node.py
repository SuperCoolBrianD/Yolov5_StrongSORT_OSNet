#! /usr/bin/env python3
# import sys
# root = '/radar_live/src'
# sys.path.append(root)
import os
root = os.path.dirname(os.path.abspath(__file__))
print(root)
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
import sys
from fusion_utils import radar_utils
import std_msgs.msg as std_msgs
from rclpy.clock import Clock
import cv2
from Yolov5_StrongSORT_OSNet.track_custom import load_weight_sort, process_track
from sensor_msgs.msg import PointCloud2


class Camera_Detection(Node):
    def __init__(self):
        super().__init__('Radar_sub')
        sys.path.append(root+'/Yolov5_StrongSORT_OSNet')
        sys.path.append(root+'/Yolov5_StrongSORT_OSNet/strong_sort/deep/reid')
        self.idx = 0
        device = '0'
        self.outputs = [None]
        self.arr_g = np.empty((0, 5))
        self.device, self.model, self.stride, self.names, self.pt, self.imgsz, self.cfg, self.strongsort_list, \
        self.dt, self.seen, self.curr_frames, self.prev_frames, self.half = load_weight_sort(device,
                                                                    root + '/Yolov5_StrongSORT_OSNet/strong_sort/configs/strong_sort.yaml')

        (rx, ry, rz, tx, ty, tz), (alpha, _, _, _, height, _), mtx, hull  = radar_utils.cfg_read(root+'/config/rooftop.yaml')
        mtx_p = np.eye(4)
        mtx_p[:3, :3] = mtx
        self.r2c_e = radar_utils.extrinsic_matrix(rx, ry, rz, tx, ty, tz)
        self.c2g = radar_utils.extrinsic_matrix(alpha, 0, 0, 0, height, 0)
        self.g2c_p = radar_utils.cam_radar(-alpha, 0, 0, 0, -height, 0, mtx)
        self.c2wcs = radar_utils.Cam2WCS(alpha, 0, 0, 0, 0, -height, mtx)                                                            
        self.camera_subscriber = self.create_subscription(Image, '/Camera', self.camera_callback, 10)
        self.radar_subscriber = self.create_subscription(PointCloud2, '/Radar', self.radar_callback, 10)
        self.camera_publisher = self.create_publisher(Image, '/Camera_detection', 10)
        
    def camera_callback(self, msg):
        image_np = radar_utils.imgmsg_to_cv2(msg)
        _, _, C_M, camera_detection, outputs = process_track(image_np, self.idx, self.curr_frames, self.prev_frames, self.outputs,
                                        self.device, self.model, self.stride, self.names, self.pt, self.imgsz, self.cfg, self.strongsort_list, self.dt,
                                        self.seen, self.half, classes=[2], conf_thres=0.75)
        img, cam_arr = radar_utils.render_radar_on_image(self.arr_g, C_M, self.g2c_p, 9000, 9000)
        cv2.imshow('0', img)
        cv2.waitKey(1)
    def radar_callback(self, msg):
        npts = msg.width
        arr_all = radar_utils.pc2_numpy(msg, npts)
        # arr_concat = np.vstack((arr_all, p_arr_all))
        arr_concat = arr_all
        arr_c = radar_utils.transform_radar(arr_concat.T, self.r2c_e).T  # from radar to camera (not pixel)
        self.arr_g = radar_utils.transform_radar(arr_c.T, self.c2g).T  # from camera to global
def camera_detection_node(args=None):
    rclpy.init(args=args)
    cam_sub = Camera_Detection()
    rclpy.spin(cam_sub)

    rclpy.shutdown()

if __name__ == '__main__':
    camera_detection_node()