#! /usr/bin/env python3
from concurrent.futures import Executor
from re import S
import sys

import os
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
print(root)
sys.path.append(root+'/Yolov5_StrongSORT_OSNet')
sys.path.append(root+'/Yolov5_StrongSORT_OSNet/strong_sort/deep/reid')
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
import sys
from fusion_utils import radar_utils
from fusion_utils import learn_utils
from fusion_utils import auto_label_util
import std_msgs.msg as std_msgs
from rclpy.clock import Clock
import cv2
from Yolov5_StrongSORT_OSNet.track_custom import load_weight_sort, process_track
from sensor_msgs.msg import PointCloud2
import matplotlib.path as mpltPath
import math
import pickle
import SVSF_Track.MTT_Functions as Track_MTT
import time
import threading
import logging
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy import qos
from Yolov5_StrongSORT_OSNet.strong_sort.deep.reid.torchreid.utils import FeatureExtractor
# from multiprocessing import Queue
from queue import Queue
from fusion_utils.detection_utils import RadarTracking
class Camera_Detection(Node):
    def __init__(self):
        super().__init__('Radar_sub')

        self.idx = 0
        device = '0'
        self.outputs = [None]
        self.arr_g = np.empty((0, 5))
        self.device, self.model, self.stride, self.names, self.pt, self.imgsz, self.cfg, self.strongsort_list, \
        self.dt, self.seen, self.curr_frames, self.prev_frames, self.half = load_weight_sort(device,
                                                                    root + '/Yolov5_StrongSORT_OSNet/strong_sort/configs/strong_sort.yaml', yolo_weights=root+'/Yolov5_StrongSORT_OSNet/weights/yolov5s.pt')
        self.svm_model = pickle.load(open(root+'/radar_model_zoo/svm_model_scale.pkl', 'rb'))
        self.scaler_model = pickle.load(open(root+'/radar_model_zoo/scaler_model.pkl', 'rb'))
        self.cls_model = pickle.load(open(root+'/person_k_means.pkl', 'rb'))
        self.cls_name = pickle.load(open(root+'/person_k_means_names.pkl', 'rb'))
        # self.extractor = FeatureExtractor(
        #     model_name='osnet_ibn_x1_0',
        #     model_path='pretrained_e.pth',
        #     device='cuda'
        # )

        
        # track parameters
        self.radar_track = RadarTracking()

        # transformation parameters
        (rx, ry, rz, tx, ty, tz), (alpha, _, _, _, height, _), mtx, hull  = radar_utils.cfg_read(root+'/config/cali.yaml')
        self.use_mask = False
        self.calib = radar_utils.calib(rx, ry, rz, tx, ty, tz, mtx, alpha, height)
        self.path = mpltPath.Path(hull)
        self.p_arr_all = np.empty((0, 5))

        # initialization parameters
        self.font_size = 0.3
        self.thickness = 1
        self.road_objects_dict = dict()
        self.road_objects_dict['no_match'] = [[], [], 0]
        self.start_list = [-1] * 8000
        self.radar_ids = [-1]
        self.tracked_object_all = []
        self.tracked_object_radar = dict()
        self.tracked_object_last = 0
        

        # buffer
        self.data_queue = Queue(maxsize=90)
        self.frame = radar_utils.SRS_data_frame_buffer_sort()
        # debugging parameters
        self.t = 0
        self.camera_time = 0
        self.radar_time = 0

        self.camera_subscriber = self.create_subscription(Image, '/Camera', self.camera_callback, 10)
        self.radar_subscriber = self.create_subscription(PointCloud2, '/Radar', self.radar_callback, 10)
        #qos.qos_profile_sensor_data)
        # self.camera_publisher = self.create_publisher(Image, '/Camera_detection', 10)
        self.camgroup = MutuallyExclusiveCallbackGroup()
        self.radgroup = MutuallyExclusiveCallbackGroup()
        self.reid = False

    def radar_callback(self, msg):
        """Populate frame with radar, check if full frame, if full frame append to data_queue"""
        # print('radar')
        self.data_queue.put((msg, 'r'))
        # if self.data_queue.qsize() > 80:
        #     self.data_queue.get()
        #     print('Warning Overflowed Frame')
        # t = radar_utils.rostime2sec(msg.header.stamp)
    def camera_callback(self, msg):
        # print('camera')
        self.data_queue.put((msg, 'c'))
        # if self.data_queue.qsize() > 80:
            # self.data_queue.get()
            # print('Warning Overflowed Frame')

        # t = radar_utils.rostime2sec(msg.header.stamp)


    def fusion(self):
        while True:
            if cv2.waitKey(0) & 0xFF == ord('x'):
                self.reid = True
            data_frame = []
            while len(data_frame) < 20:
                print(self.data_queue.qsize())
                if self.data_queue.qsize() > 40:
                    data = self.data_queue.get()
                    data_frame.append(data)
            data_frame.sort(key=radar_utils.msg_sort)
            msg_type = [i[1] for i in data_frame]
            # processing module
            for data in data_frame:
                self.frame.load_data(data)
                if self.frame.full_data:
                    camera_msg = self.frame.camera[-1]

                    image_np = radar_utils.imgmsg_to_cv2(camera_msg)
                    _, _, C_M, camera_detection, outputs = process_track(image_np, self.idx, self.curr_frames, self.prev_frames, self.outputs,
                                                    self.device, self.model, self.stride, self.names, self.pt, self.imgsz, self.cfg, self.strongsort_list, self.dt,
                                                    self.seen, self.half, conf_thres=0.5, classes=[0])
                    cam_2d = np.asarray([ii[0] for ii in camera_detection])
                    camera_ids = [ii[2] for ii in camera_detection]
                    if self.reid:
                        radar_utils.face_detection(outputs, self.extractor, self.cls_model, self.cls_name, self.names, image_np, C_M)
                    for ii in camera_ids:
                        if ii not in self.road_objects_dict.keys():
                            # index 0 is for counting radarID, index 1 is for storing index in trackList
                            self.road_objects_dict[ii] = [[], [], 0]
                        else:
                            self.road_objects_dict[ii][2] += 1
                    # print(f'recieved {len(self.frame.radar)} radar frames')
                    # if len(data.radar) > 1:
                    #     pass
                    #     # print("WARNING MULTIRADAR FRAME")
                    for i, radar_msg in enumerate(self.frame.radar):
                        npts = radar_msg.width
                        self.arr_all = radar_utils.pc2_numpy(radar_msg, npts)
                        arr_concat = np.vstack((self.arr_all, self.p_arr_all))
                        arr_c = radar_utils.transform_radar(arr_concat.T, self.calib.r2c_e).T  # from radar to camera (not pixel)
                        arr_g = radar_utils.transform_radar(arr_c.T, self.calib.c2g).T  # from camera to global
                        if self.use_mask:
                            mask = self.path.contains_points(np.vstack((arr_g[:, 0], arr_g[:, 2])).T)
                            arr = arr_g[mask]
                        else:
                            arr = radar_utils.filter_zero(arr_g, thresh=0.2)
                        pc = arr[:, :4]
                        detection_list, img, measSet = self.radar_detection(arr, C_M, pc, camera_detection, cam_2d)
                        # update next frame for radar track
                        self.radar_track.update(measSet)
                        detection_list = self.radar_track.matching(detection_list, img, self.calib)
                        # start fusion loop
                        
                        for ii in detection_list:
                            # check if this camera id then radar id is in any of the tracks
                            # ids = []
                            add_track = True
                            for jj, tk in enumerate(self.tracked_object_all):
                                # record all track with the same radar or camera ID in ids array
                                if ii.cam_id and ii.cam_id in tk.c_id:
                                    add_track = False
                                    tk.activate = ii.sensor
                                    if ii.rad_id and (ii.rad_id not in tk.r_id):
                                        # add the id if id is not in track
                                        tk.r_id.append(ii.rad_id)
                                    elif ii.rad_id and (ii.rad_id in tk.r_id):
                                        # just to maintain order and add the newest detection to the front of the list
                                        tk.r_id.append(tk.r_id.pop(tk.r_id.index(ii.rad_id)))
                                elif ii.rad_id and (ii.rad_id in tk.r_id):
                                    add_track = False
                                    tk.activate = ii.sensor
                                    if ii.cam_id and (ii.cam_id not in tk.c_id):
                                        tk.c_id.append(ii.cam_id)
                                    elif ii.cam_id and (ii.cam_id in tk.c_id):
                                        tk.c_id.append(tk.c_id.pop(tk.c_id.index(ii.cam_id)))
                                
                            if add_track:
                                if ii.cam_id or ii.rad_id:
                                    trk = radar_utils.TrackedObjectALL(c_id=ii.cam_id, r_id=ii.rad_id, sensor=ii.sensor, id=self.tracked_object_last)
                                    self.tracked_object_last+=1
                                    self.tracked_object_all.append(trk)
                            # if in_track:
                            #     # merge all the track into one track``
                            #     keep_track = ids[0]
                            #     if len(ids) > 1:
                            #         for jj in ids[1:]:
                            #             for jjj in self.tracked_object_all[jj].c_id:
                            #                 if jjj not in self.tracked_object_all[keep_track].c_id:
                            #                     self.tracked_object_all[keep_track].c_id.insert(0, jjj)
                            #             for jjj in self.tracked_object_all[jj].r_id:
                            #                 if jjj not in self.tracked_object_all[keep_track].r_id:
                            #                     self.tracked_object_all[keep_track].r_id.insert(0, jjj)
                            #             self.tracked_object_all[jj].delete()
                                # if camera id is found
                                # self.tracked_object_all[ids[0]].activate = ii.sensor
                                # add the radar id to this tracked object if not there
                        # for trk in self.tracked_object_all:
                        #     if                             # if in_track:
                            #     # merge all the track into one track``
                            #     keep_track = ids[0]
                            #     if len(ids) > 1:
                            #         for jj in ids[1:]:
                            #             for jjj in self.tracked_object_all[jj].c_id:
                            #                 if jjj not in self.tracked_object_all[keep_track].c_id:
                            #                     self.tracked_object_all[keep_track].c_id.insert(0, jjj)
                            #             for jjj in self.tracked_object_all[jj].r_id:
                            #                 if jjj not in self.tracked_object_all[keep_track].r_id:
                            #                     self.tracked_object_all[keep_track].r_id.insert(0, jjj)
                            #             self.tracked_object_all[jj].delete()
                                # if camera id is found
                                # self.tracked_object_all[ids[0]].activate = ii.sensor
                                # add the radar id to this tracked object if not there
                        # for trk in self.tracked_object_all:
                        for ii, trk in enumerate(self.tracked_object_all):
                            trk.intrack = False
                            if True:
                                if not trk.deleted and trk.activate:
                                    if trk.activate in ['Radar', 'Both', 'Radar_track'] and trk.r_id:
                                        r_id = trk.r_id
                                        r_trk = self.radar_track.trackList[r_id[-1]]
                                        centroid = r_trk.xPost[:2]
                                        speed = r_trk.speed
                                        centroid_img = np.zeros((1, 4))
                                        centroid_img[:, 0] = centroid[0]
                                        centroid_img[:, 2] = centroid[1]
                                        centroid_img[:, 1] = 0
                                        pts = radar_utils.project_to_image(centroid_img.T, self.calib.g2c_p).flatten()
                                        cv2.putText(img, f'Fusion ID: {trk.id}', (
                                            int(pts[0] + 10), int(pts[1]) - 15),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    self.font_size, (0, 255, 0), self.thickness, cv2.LINE_AA)
                                        cv2.putText(img, f'Radar_id: {r_id}', (
                                            int(pts[0] + 10), int(pts[1]) +15),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    self.font_size, (0, 255, 0), self.thickness, cv2.LINE_AA)
                                        cv2.putText(img, f'Speed: {speed:.03g}km/h', (
                                            int(pts[0]) + 10, int(pts[1]) + 3),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    self.font_size, (0, 255, 0), 1, cv2.LINE_AA)
                                        cv2.circle(img, (int(pts[0]), int(pts[1])), 5, color=(0, 255, 0), thickness=2)
                                        if trk.c_id:
                                            cv2.putText(img, f'camera_id: {trk.c_id}', (
                                                int(pts[0] + 10), int(pts[1]) + 25),
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        self.font_size, (0, 255, 0), self.thickness, cv2.LINE_AA)
                                        trk.life = 10
                                    elif trk.c_id:
                                        c_id = trk.c_id
                                        if c_id[-1] in outputs[0][:, 4]:
                                            id= np.where(outputs[0][:, 4] == c_id[-1])[0][0] # what?
                                            bbox = outputs[0][id, :4]
                                            pts = [(bbox[2] - bbox[0]) / 2 + bbox[0], bbox[3] - 10]
                                            if not trk.r_id:
                                                cv2.putText(img, f'Fusion ID: {trk.id}', (
                                                    int(pts[0] + 10), int(pts[1])),
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            self.font_size, (0, 0, 255), self.thickness, cv2.LINE_AA)
                                                cv2.circle(img, (int(pts[0]), int(pts[1])), 5, color=(0, 0, 255), thickness=2)
                                                cv2.putText(img, f'camera_id: {c_id}', (
                                                    int(pts[0] + 10), int(pts[1])+ 13),
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            self.font_size, (0, 0, 255), self.thickness, cv2.LINE_AA)
                                            if trk.r_id:
                                                r_id = trk.r_id
                                                speed = self.radar_track.trackList[r_id[-1]].speed
                                                cv2.circle(img, (int(pts[0]), int(pts[1])), 5, color=(0, 0, 255), thickness=2)
                                                cv2.putText(img, f'Fusion ID: {trk.id}', (
                                                    int(pts[0] + 10), int(pts[1])- 15),
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            self.font_size, (0, 255, 0), self.thickness, cv2.LINE_AA)
                                                cv2.putText(img, f'camera_id: {c_id}', (
                                                    int(pts[0] + 10), int(pts[1])+25),
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            self.font_size, (0, 0, 255), self.thickness, cv2.LINE_AA)
                                                cv2.putText(img, f'Speed: {speed:.03g}km/h', (
                                                    int(pts[0]) + 10, int(pts[1]) + 3),
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            self.font_size, (0, 0, 255), self.thickness, cv2.LINE_AA)
                                                cv2.putText(img, f'Radar_id: {r_id} (noD)', (
                                                    int(pts[0] + 10), int(pts[1]) + 15),
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            self.font_size, (0, 0, 255), self.thickness, cv2.LINE_AA)
                                                trk.life = 10
                                    trk.activate = False
                        cv2.imshow('0', img)
                        cv2.waitKey(10)
                        self.p_arr_all = self.arr_all.copy()
                    self.frame.clear()
                # self.data_queue.task_done()
                        # print(f'Processed Frame in {time.time()-measure_time}')
                elif self.data_queue.empty():
                    pass
                    #print('waiting for new frames')


    def radar_detection(self, arr, C_M, pc, camera_detection, cam_2d):
        detection_list = []
        total_box, cls = radar_utils.dbscan_cluster(pc, eps=2.5, min_sample=15)
        total_box_1, cls_1 = radar_utils.dbscan_cluster(pc, eps=2, min_sample=2)
        img, cam_arr = radar_utils.render_radar_on_image(arr, C_M, self.calib.g2c_p, 9000, 9000)
        if isinstance(cls, type(None)):
            cls = []
        if isinstance(cls_1, type(None)):
            cls_1 = []
        total_box = np.vstack((total_box, total_box_1))
        # print(total_box)
        box_index = radar_utils.non_max_suppression_fast(total_box[:, :4], .2)
        cls.extend(cls_1)
        cls = [cls[ii] for ii in box_index]
        total_box = total_box[box_index, :]
        measSet = np.empty((0, 4))
        if cls:
            features = np.empty((0, 12))
            radar_detection = []
            for ii, cc in enumerate(cls):
                # plot_box(total_box[ii, :], cc, axs)
                centroid = np.mean(cc, axis=0)
                # get tracking measurement
                measSet = np.vstack((measSet, centroid))
                bbox = radar_utils.get_bbox_cls(cc)
                features = np.vstack((features, np.array(learn_utils.get_features(cc, bbox))))
                bbox = radar_utils.get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)
                bbox = radar_utils.project_to_image(bbox, self.calib.g2c_p)
                pts = radar_utils.project_to_image(cc.T, self.calib.g2c_p)
                box2d = radar_utils.get_bbox_2d(pts.T)
                # cv2.rectangle(img, box2d[0], box2d[1], (255, 255, 0))
                box2d = [box2d[0][0], box2d[0][1], box2d[1][0], box2d[1][1]]
                box2d = auto_label_util.convert_topy_bottomx(box2d)
                radar_detection.append([cc, centroid, bbox, box2d])
            measSet = np.vstack((measSet[:, 0].T, measSet[:, 2].T))
            features = self.scaler_model.fit_transform(features)
            prediction = self.svm_model.predict_proba(features)
            for ii in range(len(cls)):
                pd = np.argmax(prediction[ii, :])
                radar_detection[ii].append(prediction[ii, :])

            radar_2d = np.asarray([ii[3] for ii in radar_detection])
            
            if cam_2d.any():
                radar_matched, camera_matched, ious, radar_unmatched, camera_unmatched = auto_label_util.match_detection(radar_2d, cam_2d, iou_thres=0.1)
                for ii in range(len(radar_matched)):
                    detection_list.append(radar_utils.DetectedObject(r_d=radar_detection[radar_matched[ii]], c_d=camera_detection[camera_matched[ii]]))
                for ii in radar_unmatched:
                    detection_list.append(radar_utils.DetectedObject(r_d=radar_detection[ii]))
                for ii in camera_unmatched:
                    detection_list.append(radar_utils.DetectedObject(c_d=camera_detection[ii]))
            else:
                for ii in radar_detection:
                    detection_list.append(radar_utils.DetectedObject(r_d=ii))
        elif camera_detection:
            # initiate empty radar measurement if no object is detected on radar
            # initiate all detections for camera
            measSet = np.empty((2, 0))
            for ii in camera_detection:
                detection_list.append(radar_utils.DetectedObject(c_d=ii))
        else:
            # initiate measurement if no object is detected in both camera and radar
            measSet = np.empty((2, 0))
        
        return detection_list, img, measSet





def camera_detection_node(cam_sub):
    rclpy.spin(cam_sub)
    rclpy.shutdown()

if __name__ == '__main__':
    logging.info("Initializing Subscriber Node")
    rclpy.init()
    cam_sub = Camera_Detection()
    logging.info("Starting Data Acquisition Thread")
    #x = threading.Thread(target=camera_detection_node, args=(cam_sub,))
    #x.start()
    x = threading.Thread(target=cam_sub.fusion)
    x.start()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(cam_sub)
    executor.spin()
