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
class Camera_Detection(Node):
    def __init__(self):
        super().__init__('Radar_sub')

        self.idx = 0
        device = '0'
        self.outputs = [None]
        self.arr_g = np.empty((0, 5))
        self.device, self.model, self.stride, self.names, self.pt, self.imgsz, self.cfg, self.strongsort_list, \
        self.dt, self.seen, self.curr_frames, self.prev_frames, self.half = load_weight_sort(device,
                                                                    root + '/Yolov5_StrongSORT_OSNet/strong_sort/configs/strong_sort.yaml')
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
        self.trackList = [0] * 8000  # allocate track list, a list of objects
        self.lastTrackIdx = -1  # index of a track in the list,
        # init KF tracker
        self.Ts = .05 # sampling time
        self.unassignedMeas0 = np.array([[]])

        self.modelType = 'CT' #model used for single model filters
        self.sensorType = 'Lidar'
        #filterType = "IPDAKF"
        self.filterType = "IMMIPDAKF"
        #filterType = "IMMPDAGVBLSVSF"
        # filterType = "IPDASVSF"
        self.sensorPos = np.array([0, 0])

        if self.filterType=="IMMIPDAKF" or self.filterType == "IMMIPDAGVBLSVSF":
            self.isMM = True
        else:
            self.isMM = False

        # new for svsf

        self.n = 7 # x, y, vx, vy, ax, ay, turn-rate
        self.m = 2 #number of measurements
        self.psi1 = 1000 # p larger uncertainty increase
        self.psi2 = 1000 # v larger uncertainty increase
        self.psi3 = 10 # a larger uncertainty increase
        self.psi4 = 10 # turn rate larger uncertainty increase
        self.gammaZ = .1 * np.eye(self.m) # convergence rate stability from 0-1 for measured state
        self.gammaY = .1 * np.eye(self.n - self.m) # for unmeasured state

        self.psiZ = np.array([self.psi1, self.psi1])
        self.psiY = np.array([self.psi2, self.psi2, self.psi3, self.psi3, self.psi4])
        self.T_mat = np.eye(self.n)

        self.SVSFParams = [self.psiZ,self.psiY,self.gammaZ,self.gammaY,self.T_mat]

        #Standard deviations for process and measurement noises
        self.sigma_v = 1E-1#process noise standard deviation
        self.sigma_v_filt = self.sigma_v #process noise standard deviation for filter
        self.sigma_w = .5 #measurement noise standard deviation in position

        # Process noise covariances
        self.Q_CV = np.diag([self.sigma_v**2, self.sigma_v**2,0]) #process noise co-variance for CV model
        self.Q_CA = np.diag([self.sigma_v**2, self.sigma_v**2,0]) #process noise co-variance for CA model
        self.Q_CT = np.diag([self.sigma_v**2, self.sigma_v**2, (5*self.sigma_v)**2]) #process noise co-variance for CT model

        self.R = np.diag(np.array([self.sigma_w ** 2, self.sigma_w ** 2]))  # measurement co-variance
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]])  # measurement matrix

        #Input gain matrices
        self.G_CV = np.array([[(self.Ts**2)/2, 0,0],[0, (self.Ts**2)/2,0],[self.Ts,0,0],[0,self.Ts,0],[0,0,0],[0,0,0],[0,0,0]])  #input gain for CV
        self.G_CA = np.array([[(self.Ts**2)/2, 0,0],[0, (self.Ts**2)/2,0],[self.Ts,0,0],[0,self.Ts,0], [1,0,0],[0,1,0],[0,0,0]]) #input gain for CA
        self.G_CT = np.array([[(self.Ts**2)/2, 0,0],[0, (self.Ts**2)/2,0],[self.Ts,0,0],[0,self.Ts,0],[0,0,0],[0,0,0],[0,0,self.Ts]]) #input gain for CT

        #Parameters for 1-point initialization
        self.maxAcc = 2
        self.maxVel = 25 #for initializing velocity variance for 1-point initialization
        self.omegaMax = math.radians(4) #for initializing turn-rate variance for 1-point initialization
        self.maxVals = [self.maxVel, self.maxAcc, self.omegaMax]

        #Parameters for IPDA
        self.pInit = .4 #initial probability of track existence
        self.PD = .6 #probability of target detection in a time step
        #PG = .99999 #gate probability
        self.PG = .99999
        self.lambdaVal = 0.05 # parameter for clutter density

        self.useLogic= False
        self.delTenThr = .05/50 #threshold for deleting a tenative track
        self.confTenThr = .6 # threshold for confirming a tenative track/
        self.delConfThr = 0.01/10 # threshold for deleting a confirmed track
        self.N = 10000
        self.k = 0
        
        #IMM parameters
        self.models = ["CV", 'CA','CT']
        # models = ["CV"]
        self.filters = ['IPDAKF', 'IPDAKF','IPDAKF']
        # filters = ['IPDAKF']
        self.G_List = [self.G_CV, self.G_CA,self.G_CT] #input gain list
        self.Q_List = [self.Q_CV, self.Q_CA, self.Q_CT] #process noise co-variance list
        #pInits = [.2,.2] #initial track existence probabilities
        #uVec0 = [.5, .5] #initial mode probabilities
        self.r = len(self.models)
        #Set Markov matrix for IMM below
        self.P_ii_IMM = .99
        self.P_ij_IMM = (1-self.P_ii_IMM)/self.r
        if self.r==2:
            self.MP_IMM = np.array([[self.P_ii_IMM, self.P_ij_IMM], [self.P_ij_IMM, self.P_ii_IMM]])
        elif self.r==3:
            self.MP_IMM = np.array([[self.P_ii_IMM,self.P_ij_IMM,self.P_ij_IMM],[self.P_ij_IMM,self.P_ii_IMM,self.P_ij_IMM],[self.P_ij_IMM,self.P_ij_IMM,self.P_ii_IMM]])


        P_ii = .999 #for Markov matrix of IPDA
        self.MP_IPDA = np.array([[P_ii, 1-P_ii], [1-P_ii, P_ii]]) #Markov matrix for IPDA
        

        if self.modelType == "CV":
            self.Q=self.Q_CV
            self.G=self.G_CV
        elif self.modelType == "CA":
            self.Q=self.Q_CA
            self.G=self.G_CA
        elif self.modelType=="CT":
            self.Q=self.Q_CT
            self.G=self.G_CT

        #For M/N logic based track management
        self.N1 = 3
        self.M2 = 10
        self.N2 = 12
        self.N3 = 6

        # transformation parameters
        (rx, ry, rz, tx, ty, tz), (alpha, _, _, _, height, _), mtx, hull  = radar_utils.cfg_read(root+'/config/cali.yaml')
        self.use_mask = False
        self.mtx_p = np.eye(4)
        self.mtx_p[:3, :3] = mtx
        self.r2c_e = radar_utils.extrinsic_matrix(rx, ry, rz, tx, ty, tz)
        self.c2g = radar_utils.extrinsic_matrix(alpha, 0, 0, 0, height, 0)
        self.g2c_p = radar_utils.cam_radar(-alpha, 0, 0, 0, -height, 0, mtx)
        self.c2wcs = radar_utils.Cam2WCS(alpha, 0, 0, 0, 0, -height, mtx)
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
        self.tracked_list = [radar_utils.RadarTrackedObject() for i in range(8000)]
        self.d_weight = np.array([1.25998060e-01, 0.00000000e+00, 3.72307987e-04, 2.97648267e-04,
                     4.07682852e-03, 3.03747762e-03, 2.07011840e-02, 2.03772601e-01,
                     5.00000000e-01])

        # buffer
        self.data_queue = []
        self.frame = radar_utils.SRS_data_frame_buffer()
        # debugging parameters
        self.t = 0
        self.camera_time = 0
        self.radar_time = 0

        self.camera_subscriber = self.create_subscription(Image, '/Camera', self.camera_callback, 1)
        self.radar_subscriber = self.create_subscription(PointCloud2, '/Radar', self.radar_callback, 1)
        # self.camera_publisher = self.create_publisher(Image, '/Camera_detection', 10)
        self.camgroup = MutuallyExclusiveCallbackGroup()
        self.radgroup = MutuallyExclusiveCallbackGroup()
        self.frame = radar_utils.SRS_data_frame()
    def radar_callback(self, msg):
        """Populate frame with radar, check if full frame, if full frame append to data_queue"""
        # print('radar')
        self.frame.load_data_radar(msg)
    def camera_callback(self, msg):
        # print('camera')
        self.frame.load_data_camera(msg)
            
    def fusion(self):
        while True:
            if self.frame.full_data:
                measure_time = time.time()
                data = self.frame
                camera_msg = data.camera
                image_np = radar_utils.imgmsg_to_cv2(camera_msg)
                _, _, C_M, camera_detection, outputs = process_track(image_np, self.idx, self.curr_frames, self.prev_frames, self.outputs,
                                                self.device, self.model, self.stride, self.names, self.pt, self.imgsz, self.cfg, self.strongsort_list, self.dt,
                                                self.seen, self.half, conf_thres=0.5, classes=[0])
                cam_2d = np.asarray([ii[0] for ii in camera_detection])
                camera_ids = [ii[2] for ii in camera_detection]
                # radar_utils.face_detection(outputs, self.extractor, self.cls_model, self.cls_name, self.names, image_np, C_M)
                for i, radar_msg in enumerate(data.radar):

                    npts = radar_msg.width
                    self.arr_all = radar_utils.pc2_numpy(radar_msg, npts)
                    arr_concat = np.vstack((self.arr_all, self.p_arr_all))
                    arr_c = radar_utils.transform_radar(arr_concat.T, self.r2c_e).T  # from radar to camera (not pixel)
                    arr_g = radar_utils.transform_radar(arr_c.T, self.c2g).T  # from camera to global
                    if self.use_mask:
                        mask = self.path.contains_points(np.vstack((arr_g[:, 0], arr_g[:, 2])).T)
                        arr = arr_g[mask]
                    else:
                        arr = radar_utils.filter_zero(arr_g)
                    pc = arr[:, :4]
                    detection_list = []
                    total_box, cls = radar_utils.dbscan_cluster(pc, eps=2.5, min_sample=15)
                    total_box_1, cls_1 = radar_utils.dbscan_cluster(pc, eps=2, min_sample=2)
                    img, cam_arr = radar_utils.render_radar_on_image(arr, C_M, self.g2c_p, 9000, 9000)
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
                            bbox = radar_utils.project_to_image(bbox, self.g2c_p)
                            pts = radar_utils.project_to_image(cc.T, self.g2c_p)
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
                            radar_matched, camera_matched, ious, radar_unmatched, camera_unmatched = auto_label_util.match_detection(radar_2d, cam_2d)
                            for ii in range(len(radar_matched)):
                                display = image_np.copy()
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
                    tracking_time = time.time()
                    self.trackList, self.unassignedMeas = Track_MTT.gating(self.trackList, self.lastTrackIdx, self.PG, self.MP_IMM, self.maxVals, self.sensorPos, measSet, self.k)
                    # perform gating
                    self.trackList = Track_MTT.updateStateTracks(self.trackList, self.lastTrackIdx, self.filterType, measSet, self.maxVals,
                                                self.lambdaVal, self.MP_IPDA, self.PG, self.PD, self.sensorPos, self.SVSFParams, self.k)
                    # update the state of each track
                    if self.useLogic == True:
                        self.trackList = Track_MTT.updateTracksStatus_MN(self.trackList, self.lastTrackIdx, self.N1, self.M2, self.N2, self.N3, self.k)
                    else:
                        self.trackList = Track_MTT.updateTracksStatus(self.trackList, self.lastTrackIdx, self.delTenThr, self.delConfThr, self.confTenThr,
                                                    self.k)  # update the status of each track usiing the track manager

                    # update the status of each track usiing the track manager
                    # initiate tracks for measurements that were not gated or in other words unassigned measurements

                    if self.isMM == True:
                        self.trackList, self.lastTrackIdx = Track_MTT.initiateTracksMM(self.trackList, self.lastTrackIdx, self.unassignedMeas, self.maxVals, self.G_List, self.H,
                                                                self.Q_List, self.R, self.models, self.filters, self.Ts, self.pInit, self.k, self.sensorType,
                                                                self.sensorPos, self.N)
                    else:
                        self.trackList, self.lastTrackIdx = Track_MTT.initiateTracks(self.trackList, self.lastTrackIdx, measSet, self.maxVel, self.maxAcc, self.omegaMax, self.G,
                                                                self.H, self.Q, self.R, self.modelType, self.Ts, self.pInit, self.k, self.sensorType, self.sensorPos, self.N)
                    # print(f"tracking time = {time.time()- tracking_time}")
                    self.k+=1
                    matched_cameras = []
                    print('fk i died')
                    for jj, ii in enumerate(self.trackList[:self.lastTrackIdx]):
                        if self.lastTrackIdx == -1:
                            break
                        # get centroid
                        centroid = ii.xPost[:2]
                        if not ii.endSample:
                            # axs.text(centroid[0], centroid[1] - 2, 'ID: ' + str(jj), fontsize=11,
                            #          color='r')
                            # axs.scatter(centroid[0], centroid[1], s=5, color='r')
                            # if jj in [4, 16, 29 ,33]:
                            #     axs.text(centroid[0], centroid[1] - 2, 'ID: ' + str(jj), fontsize=11,
                            #              color='r')
                            #     axs.scatter(centroid[0], centroid[1], s=5, color='r')
                            speed = np.sqrt(ii.xPost[2] ** 2 + ii.xPost[3] ** 2) * 3.6
                            ii.speed = speed
                            v = ii.xPost[3], ii.xPost[2]
                            angle = math.atan2(ii.xPost[3], ii.xPost[2])
                            x1 = math.cos(angle) * 5 + centroid[0]
                            y1 = math.sin(angle) * 5 + centroid[1]
                            latency = ii.latency
                            pCurrent = ii.pCurrent
                            status = ii.status
                            # if track has terminated remove from tracked objects
                            centroid_img = np.zeros((1, 4))
                            centroid_img[:, 0] = centroid[0]
                            centroid_img[:, 2] = centroid[1]
                            centroid_img[:, 1] = 0
                            pts = radar_utils.project_to_image(centroid_img.T, self.g2c_p).flatten()
                            cv2.circle(img, (int(pts[0]), int(pts[1])), 5, color=(0, 255, 0), thickness=2)
                            # s_img = np.zeros((1, 4))
                            # s_img[:, :2] = [x1, y1]
                            # s_img[:, 2] = -2
                            # s_pts = radar_utils.project_to_image(s_img.T, g2c_p).flatten()
                            matched_measurement = radar_utils.match_measurement(detection_list, centroid)
                            # if jj == 25:
                            #     print(ii.c_ID)
                            if matched_measurement != None:
                                # if at the detection stage a camera was matched with the radar
                                if detection_list[matched_measurement].cam_id:
                                    ii.c_ID.append(detection_list[matched_measurement].cam_id)

                                self.tracked_list[jj].dets.append(detection_list[matched_measurement])
                                self.tracked_list[jj].start = self.k
                                c_track_id = None
                                if ii.c_ID:
                                    # the current radar track and camera ID
                                    if len(ii.c_ID) > 5:
                                        c_track_id = max(ii.c_ID[-5:], key=ii.c_ID.count)
                                    else:
                                        c_track_id = max(ii.c_ID, key=ii.c_ID.count)
                                    # mark the matched camera ID
                                    matched_cameras.append(c_track_id)
                                detection_list[matched_measurement].rad_id = jj
                                if not detection_list[matched_measurement].cam_id:
                                    detection_list[matched_measurement].cam_id = c_track_id
                            else:
                                c_track_id = None
                                if ii.c_ID:
                                    # the current radar track and camera ID
                                    if len(ii.c_ID) > 5:
                                        c_track_id = max(ii.c_ID[-5:], key=ii.c_ID.count)
                                    else:
                                        c_track_id = max(ii.c_ID, key=ii.c_ID.count)
                                    # mark the matched camera ID
                                    matched_cameras.append(c_track_id)
                                trk = radar_utils.DetectedObject(trk=[jj, centroid, c_track_id])
                                
                                detection_list.append(trk)
                                # If this radar ID isn't matched with the camera previously add it
                            # axs.plot([centroid[0], x1], [centroid[1], y1], linewidth=2, markersize=5)
                            # c_sx = s_pts[0] - pts[0]
                            # c_sy = s_pts[1] - pts[1]
                            # angle = math.atan2(c_sy, c_sx)
                            # c_x1 = math.cos(angle) * 40 + pts[0]
                            # c_y1 = math.sin(angle) * 40 + pts[1]
                            # cv2.line(im0, (int(pts[0]), int(pts[1])), (int(c_x1), int(c_y1)), color=(0, 255, 0), thickness=2)
                            if self.tracked_list[jj].dets:
                                radar_label, centroid_track, num_pts, cam_label, cam_box, cam_id = self.tracked_list[jj].get_prediction()
                                track_label = learn_utils.distance_weighted_voting_custom(radar_label, centroid_track, self.d_weight)
                                ii.label = track_label
                        elif ii.endSample:
                            # gather info about the track including camera ID
                            # radar_label, r_centroid, num_pts, cam_label, cam_box, cam_id = self.tracked_list[jj].get_prediction()
                            # track_label = learn_utils.distance_weighted_voting_custom(radar_label, r_centroid, self.d_weight)
                            # axs.plot(centroid[0], centroid[1], marker="o", markersize=5)
                            # print(jj)
                            # print(tracked_list[jj].start)
                            # tracked_object[jj] = None
                            continue
                        # add if first appearance
                        if jj not in self.tracked_object_radar.keys():
                            self.tracked_object_radar[jj] = radar_utils.radar_object((centroid[0], centroid[1]))
                        # update if in tracked objects
                        elif jj in self.tracked_object_radar.keys() and self.tracked_object_radar[jj]:
                            self.tracked_object_radar[jj].upd((centroid[0], centroid[1]))
                    # end radar tracking loop
                    # start fusion loop 
                    for ii in detection_list:
                        # check if this camera id then radar id is in any of the tracks
                        in_track = False
                        ids = []
                        # if ii.rad_id == 28:
                        #     print(ii.cam_id)
                        for jj, tk in enumerate(self.tracked_object_all):
                            # record all track with the same radar or camera ID in ids array
                            if not tk.deleted:
                                if ii.cam_id and ii.cam_id in tk.c_id:
                                    in_track = True
                                    ids.append(jj)
                                    if ii.rad_id and ii.rad_id not in tk.r_id:
                                        tk.r_id.append(ii.rad_id)
                                    elif ii.rad_id and ii.rad_id in tk.r_id:
                                        tk.r_id.append(tk.r_id.pop(tk.r_id.index(ii.rad_id)))
                                elif ii.rad_id and ii.rad_id in tk.r_id:
                                    in_track = True
                                    ids.append(jj)
                                    if ii.cam_id and ii.cam_id not in tk.c_id:
                                        tk.c_id.append(ii.cam_id)
                                    elif ii.cam_id and ii.cam_id in tk.c_id:
                                        tk.c_id.append(tk.c_id.pop(tk.c_id.index(ii.cam_id)))
                        if in_track:
                            # merge all the track into one track``
                            keep_track = ids[0]
                            if len(ids) > 1:
                                for jj in ids[1:]:
                                    for jjj in self.tracked_object_all[jj].c_id:
                                        if jjj not in self.tracked_object_all[keep_track].c_id:
                                            self.tracked_object_all[keep_track].c_id.insert(0, jjj)
                                    for jjj in self.tracked_object_all[jj].r_id:
                                        if jjj not in self.tracked_object_all[keep_track].r_id:
                                            self.tracked_object_all[keep_track].r_id.insert(0, jjj)
                                    self.tracked_object_all[jj].delete()
                            # if camera id is found
                            self.tracked_object_all[ids[0]].activate = ii.sensor
                            # add the radar id to this tracked object if not there
                        elif ii.cam_id or ii.rad_id:
                            trk = radar_utils.TrackedObjectALL(c_id=ii.cam_id, r_id=ii.rad_id, sensor=ii.sensor)
                            self.tracked_object_all.append(trk)
                    for ii, trk in enumerate(self.tracked_object_all):
                        if True:
                            if not trk.deleted and trk.activate:
                                if trk.activate in ['Radar', 'Both', 'Radar_track'] and trk.r_id:
                                    r_id = trk.r_id
                                    r_trk = self.trackList[r_id[-1]]
                                    centroid = r_trk.xPost[:2]
                                    speed = r_trk.speed
                                    centroid_img = np.zeros((1, 4))
                                    centroid_img[:, 0] = centroid[0]
                                    centroid_img[:, 2] = centroid[1]
                                    centroid_img[:, 1] = 0
                                    pts = radar_utils.project_to_image(centroid_img.T, self.g2c_p).flatten()
                                    cv2.putText(img, f'Fusion ID: {ii}', (
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
                                            cv2.putText(img, f'Fusion ID: {ii}', (
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
                                            speed = self.trackList[r_id[-1]].speed
                                            cv2.circle(img, (int(pts[0]), int(pts[1])), 5, color=(0, 0, 255), thickness=2)
                                            cv2.putText(img, f'Fusion ID: {ii}', (
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
                    cv2.waitKey(1)
                    self.p_arr_all = self.arr_all.copy()
                    # print(f'Processed Frame in {time.time()-measure_time}')
                self.frame.clear_data()





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
