import SVSF_Track.MTT_Functions as Track_MTT
import cv2
from fusion_utils import radar_utils, learn_utils, auto_label_util
import numpy as np
import math

class RadarTracking:

    def __init__(self):
        m = 100000
        self.trackList = [0] * m  # allocate track list, a list of objects
        
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
        self.N = m
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
            self.G=self.G_CAearance
        self.M2 = 10
        self.N2 = 12
        self.N3 = 6
        # classifcation distance weight
        self.d_weight = np.array([1.25998060e-01, 0.00000000e+00, 3.72307987e-04, 2.97648267e-04,
                4.07682852e-03, 3.03747762e-03, 2.07011840e-02, 2.03772601e-01,
                5.00000000e-01])
        self.tracked_list = [radar_utils.RadarTrackedObject() for i in range(8000)]
    def update(self, measSet):
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
        self.k+=1

    def matching(self, detection_list, img, calib):
        matched_cameras = []
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
                # get heading angle
                angle = math.atan2(ii.xPost[3], ii.xPost[2])
                x1 = math.cos(angle) * 5 + centroid[0]
                y1 = math.sin(angle) * 5 + centroid[1]
                latency = ii.latency
                pCurrent = ii.pCurrent
                status = ii.status
                # draw track centroid on image
                centroid_img = np.zeros((1, 4))
                centroid_img[:, 0] = centroid[0]
                centroid_img[:, 2] = centroid[1]
                # assume a z of zero
                centroid_img[:, 1] = 0
                pts = radar_utils.project_to_image(centroid_img.T, calib.g2c_p).flatten()
                cv2.circle(img, (int(pts[0]), int(pts[1])), 5, color=(0, 255, 0), thickness=2)
                # old code for projecting heading image
                # s_img = np.zeros((1, 4))
                # s_img[:, :2] = [x1, y1]
                # s_img[:, 2] = -2
                # s_pts = radar_utils.project_to_image(s_img.T, g2c_p).flatten()
                # todo no long needed once bounding box information is encoded in tracking 
                matched_measurement = radar_utils.match_measurement(detection_list, centroid) # match track back to detection to find bounding box

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
                    detection_list[matched_measurement].rad_id = jj
                    if not detection_list[matched_measurement].cam_id:
                        detection_list[matched_measurement].cam_id = c_track_id
                    trk = radar_utils.DetectedObject(trk=[jj, centroid, c_track_id])
                    detection_list.append(trk)
                    # If this radar ID isn't matched with the camera previously add it
                if self.tracked_list[jj].dets:
                    # classify the track using distance based voting
                    radar_label, centroid_track, num_pts, cam_label, cam_box, cam_id = self.tracked_list[jj].get_prediction()
                    track_label = learn_utils.distance_weighted_voting_custom(radar_label, centroid_track, self.d_weight)
                    ii.label = track_label
            # add if first appearance
        # end radar tracking loop
        return detection_list


def radar_detection(arr, C_M, pc, camera_detection, cam_2d, calib_g2c_p, scaler_model, svm_model):
    detection_list = []
    total_box, cls = radar_utils.dbscan_cluster(pc, eps=2, min_sample=5)
    total_box_1, cls_1 = radar_utils.dbscan_cluster(pc, eps=1.5, min_sample=2)
    img, cam_arr = radar_utils.render_radar_on_image(arr, C_M, calib_g2c_p, 9000, 9000)
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
            bbox = radar_utils.project_to_image(bbox, calib_g2c_p)
            pts = radar_utils.project_to_image(cc.T, calib_g2c_p)
            box2d = radar_utils.get_bbox_2d(pts.T)
            cv2.rectangle(img, box2d[0], box2d[1], (255, 255, 0))
            box2d = [box2d[0][0], box2d[0][1], box2d[1][0], box2d[1][1]]
            box2d = auto_label_util.convert_topy_bottomx(box2d)
            radar_detection.append([cc, centroid, bbox, box2d])
        measSet = np.vstack((measSet[:, 0].T, measSet[:, 2].T))
        features = scaler_model.fit_transform(features)
        prediction = svm_model.predict_proba(features)
        for ii in range(len(cls)):
            pd = np.argmax(prediction[ii, :])
            radar_detection[ii].append(prediction[ii, :])

        radar_2d = np.asarray([ii[3] for ii in radar_detection])

        if cam_2d.any():
            radar_matched, camera_matched, ious, radar_unmatched, camera_unmatched = auto_label_util.match_detection(
                radar_2d, cam_2d)
            for ii in range(len(radar_matched)):
                detection_list.append(radar_utils.DetectedObject(r_d=radar_detection[radar_matched[ii]],
                                                                 c_d=camera_detection[camera_matched[ii]]))
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

# def update(measSet,trackList, lastTrackIdx, PG, MP_IMM, maxVals, sensorPos, k, filterType, lambdaVal, MP_IPDA, PD, SVSFParams, useLogic, N1, M2, N2, N3, delTenThr, delConfThr, confTenThr,
#            isMM, G_List, H, Q_List, R, models, filters, Ts, pInit, sensorType, maxVel, maxAcc, omegaMax, G):
#     trackList, unassignedMeas = Track_MTT.gating(trackList, lastTrackIdx, PG, MP_IMM, maxVals, sensorPos, measSet, k)
#     # perform gating
#     trackList = Track_MTT.updateStateTracks(trackList, lastTrackIdx, filterType, measSet, maxVals,
#                                 lambdaVal, MP_IPDA, PG, PD, sensorPos, SVSFParams, k)
#     # update the state of each track
#     if useLogic == True:
#         trackList = Track_MTT.updateTracksStatus_MN(trackList, lastTrackIdx, N1, M2, N2, N3, k)
#     else:
#         trackList = Track_MTT.updateTracksStatus(trackList, lastTrackIdx, delTenThr, delConfThr, confTenThr,
#                                     k)  # update the status of each track usiing the track manager
#
#     # update the status of each track usiing the track manager
#     # initiate tracks for measurements that were not gated or in other words unassigned measurements
#
#     if isMM == True:
#         trackList, lastTrackIdx = Track_MTT.initiateTracksMM(trackList, lastTrackIdx, unassignedMeas, maxVals, G_List, H,
#                                                 Q_List, R, models, filters, Ts, pInit, k, sensorType,
#                                                 sensorPos, N)
#     else:
#         trackList, lastTrackIdx = Track_MTT.initiateTracks(trackList, lastTrackIdx, measSet, maxVel, maxAcc, omegaMax, G,
#                                                 self.H, self.Q, self.R, self.modelType, self.Ts, self.pInit, self.k, self.sensorType, self.sensorPos, self.N)
#     self.k+=1
