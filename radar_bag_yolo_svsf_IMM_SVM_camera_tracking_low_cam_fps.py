import sys
sys.path.append('SVSF_Track')
run_cam_d = True
if run_cam_d:
    # sys.path.append('yolor')
    # from yolor.detect_custom import init_yoloR, detect
    # model, device, colors, names = init_yoloR(weights='yolor/yolor_p6.pt', cfg='yolor/cfg/yolor_p6.cfg',
    #                                            names='yolor/data/coco.names', out='inference/output', imgsz=1280, half=half)
    from Yolov5_StrongSORT_OSNet.track_custom import load_weight_sort, process_track
    sys.path.append('Yolov5_StrongSORT_OSNet')
    sys.path.append('Yolov5_StrongSORT_OSNet/strong_sort/deep/reid')
    device = '0'
    outputs = [None]
    device, model, stride, names, pt, imgsz, cfg, strongsort_list, \
    dt, seen, curr_frames, prev_frames, half = load_weight_sort(device,
                                                                'Yolov5_StrongSORT_OSNet/strong_sort/configs/strong_sort.yaml')

    print(names)

from radar_utils import *
import rosbag
from SVSF_Track.MTT_Functions import *
from SVSF_Track.track import track
import pickle
from matplotlib.animation import FuncAnimation
from learn_util import *
from auto_label_util import *
import matplotlib
matplotlib.use('TkAgg')
import time
# Read recording

bag = rosbag.Bag("record/remote.bag")
# bag = rosbag.Bag("record/traffic3.bag")
# bag = rosbag.Bag("record/traffic1.bag")
topics = bag.get_type_and_topic_info()

# old SORT tracker
# mot_tracker = sort.Sort(min_hits=2, max_age=8, iou_threshold=0.1)

# init plt figure
fig, axs = plt.subplots(1, figsize=(6, 6))
fig.canvas.set_window_title('Radar Detection and Tracking IMM_small')
# create generator object for recording
bg = bag.read_messages()
s = 0

# adjust image visualization
cv2.imshow('Camera', np.zeros((480, 640)))
cv2.moveWindow('Camera', 800, 800)
image_np = np.empty((5, 5))
# init loop
radar_time_0 = 0
camera_time_0 = 0
update = 1
cam1 = np.empty((5,5))
tracked_object_radar = dict()
tracked_object_all = []
alive_track = [0] * 8000
life = 10
# track parameters
trackList = [0] * 8000  # allocate track list, a list of objects
lastTrackIdx = -1  # index of a track in the list,
# init KF tracker


Ts = .05 # sampling time
unassignedMeas0 = np.array([[]])

modelType = 'CT' #model used for single model filters
sensorType = 'Lidar'
#filterType = "IPDAKF"
filterType = "IMMIPDAKF"
#filterType = "IMMPDAGVBLSVSF"
# filterType = "IPDASVSF"
sensorPos = np.array([0, 0])

if filterType=="IMMIPDAKF" or filterType == "IMMIPDAGVBLSVSF":
    isMM = True
else:
    isMM = False

# new for svsf

n = 7 # x, y, vx, vy, ax, ay, turn-rate
m = 2 #number of measurements
psi1 = 1000 # p larger uncertainty increase
psi2 = 1000 # v larger uncertainty increase
psi3 = 10 # a larger uncertainty increase
psi4 = 10 # turn rate larger uncertainty increase
gammaZ = .1 * np.eye(m) # convergence rate stability from 0-1 for measured state
gammaY = .1 * np.eye(n - m) # for unmeasured state

psiZ = np.array([psi1, psi1])
psiY = np.array([psi2, psi2, psi3, psi3, psi4])
T_mat = np.eye(n)

SVSFParams = [psiZ,psiY,gammaZ,gammaY,T_mat]

#Standard deviations for process and measurement noises
sigma_v = 1E-1#process noise standard deviation
sigma_v_filt = sigma_v #process noise standard deviation for filter
sigma_w = .5 #measurement noise standard deviation in position

# Process noise covariances
Q_CV = np.diag([sigma_v**2,sigma_v**2,0]) #process noise co-variance for CV model
Q_CA = np.diag([sigma_v**2,sigma_v**2,0]) #process noise co-variance for CA model
Q_CT = np.diag([sigma_v**2,sigma_v**2, (5*sigma_v)**2]) #process noise co-variance for CT model

R = np.diag(np.array([sigma_w ** 2, sigma_w ** 2]))  # measurement co-variance
H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]])  # measurement matrix

#Input gain matrices
G_CV = np.array([[(Ts**2)/2, 0,0],[0, (Ts**2)/2,0],[Ts,0,0],[0,Ts,0],[0,0,0],[0,0,0],[0,0,0]])  #input gain for CV
G_CA = np.array([[(Ts**2)/2, 0,0],[0, (Ts**2)/2,0],[Ts,0,0],[0,Ts,0], [1,0,0],[0,1,0],[0,0,0]]) #input gain for CA
G_CT = np.array([[(Ts**2)/2, 0,0],[0, (Ts**2)/2,0],[Ts,0,0],[0,Ts,0],[0,0,0],[0,0,0],[0,0,Ts]]) #input gain for CT

#Parameters for 1-point initialization
maxAcc = 2
maxVel = 25 #for initializing velocity variance for 1-point initialization
omegaMax = math.radians(4) #for initializing turn-rate variance for 1-point initialization
maxVals = [maxVel, maxAcc, omegaMax]

#Parameters for IPDA
pInit = .4 #initial probability of track existence
PD = .6 #probability of target detection in a time step
#PG = .99999 #gate probability
PG = .99999
lambdaVal = 0.05 # parameter for clutter density

useLogic= False
delTenThr = .05/50 #threshold for deleting a tenative track
confTenThr = .6 # threshold for confirming a tenative track/
delConfThr = 0.01/10 # threshold for deleting a confirmed track

#For M/N logic based track management
N1 = 3
M2 = 10
N2 = 12
N3 = 6

#IMM parameters
models = ["CV", 'CA','CT']
# models = ["CV"]
filters = ['IPDAKF', 'IPDAKF','IPDAKF']
# filters = ['IPDAKF']
G_List = [G_CV, G_CA,G_CT] #input gain list
Q_List = [Q_CV, Q_CA,Q_CT] #process noise co-variance list
#pInits = [.2,.2] #initial track existence probabilities
#uVec0 = [.5, .5] #initial mode probabilities
r = len(models)

#Set Markov matrix for IMM below
P_ii_IMM = .99
P_ij_IMM = (1-P_ii_IMM)/r
if r==2:
    MP_IMM = np.array([[P_ii_IMM, P_ij_IMM], [P_ij_IMM, P_ii_IMM]])
elif r==3:
    MP_IMM = np.array([[P_ii_IMM,P_ij_IMM,P_ij_IMM],[P_ij_IMM,P_ii_IMM,P_ij_IMM],[P_ij_IMM,P_ij_IMM,P_ii_IMM]])


P_ii = .999 #for Markov matrix of IPDA
MP_IPDA = np.array([[P_ii, 1-P_ii], [1-P_ii, P_ii]]) #Markov matrix for IPDA


if modelType == "CV":
    Q=Q_CV
    G=G_CV
elif modelType == "CA":
    Q=Q_CA
    G=G_CA
elif modelType=="CT":
    Q=Q_CT
    G=G_CT


p_time = 0
N = 10000
# trackList,lastTrackIdx = initiateTracks(trackList,lastTrackIdx, unassignedMeas0, maxVel, omegaMax, G, H, Q, R,
#                                         modelType, Ts, pInit, 0, sensor, N)

# trackList, lastTrackIdx = initiateTracksMM(trackList,lastTrackIdx, unassignedMeas0, maxVals, G_List, H, Q_List, R,
#                                            models, filters, Ts, pInit, 0, sensor, N)
k = 0
radar_p_time = 0
t_array = []
dt_array = []
dt_no_match_array = []


mtx = np.array([[748, 0., 655.5],
                [0.,746.6,390.11],
                [0., 0., 1.]])
cam_msg = 0
# mtx = np.array([[234.45076996, 0., 334.1804498],
#                 [0.,311.6748573,241.50825294],
#                 [0., 0., 1.]])
svm_model = pickle.load(open('svm_model_scale.pkl', 'rb'))
scaler_model = pickle.load(open('scaler_model.pkl', 'rb'))
classes = ['car', 'bus', 'person', 'truck', 'no_match']
yolo_classes = ['car', 'bus', 'person', 'truck', 'no_match']
# 1.56 0.0 0.05999999999999983 0.7000000000000002 0.0 0.0
# rx = 1.6
# ry = 0
# rz = .04
# tx = 0
# ty = 0
# tz = 0

# for rooftop
#
# rx = 1.58
# ry = 0
# rz = 0.05
# tx = 0.2999999999999998
# ty = 0
# tz = 0
#height = 12.5
# alpha = 22/180*np.pi

# for remote dark
rx = 1.6
ry = 0
rz = 0.04
tx = 0
ty = 0
tz = 0
mtx_p = np.eye(4)
mtx_p[:3, :3] = mtx
height = 10.5
alpha = (90-64)/180*np.pi
r2c_e = extrinsic_matrix(rx, ry, rz, tx, ty, tz)
c2g = extrinsic_matrix(-alpha, 0, 0, 0, -height, 0)
g2c_p = cam_radar(alpha, 0, 0, 0, height, 0, mtx)
c2wcs = Cam2WCS(-alpha, 0, 0, 0, 0, height, mtx)
frame = SRS_data_frame()
idx=0
epoch = 0
cluster_hist = [[] for i in range(8000)]
class_track = [[] for i in range(8000)]
tracked_list = [RadarTrackedObject() for i in range(8000)]
bus_count = 0
person_count = 0
car_count = 0
truck_count = 0



# for tripod
# hull = np.array([[-18.06451613,  17.53246753],
#        [ 53.87096774, 114.93506494],
#        [ 63.87096774, 102.81385281],
#        [ 14.51612903,  27.92207792],
#        [ 10.96774194,   9.74025974],
#        [ -1.61290323,  -1.94805195],
#         [-18.06451613,  17.53246753]])
# for working
# hull = np.array([[ 16.12903226, 118.83116883],
#        [-16.77419355,  23.59307359],
#        [ -0.64516129,  16.23376623],
#        [ 23.87096774,  87.66233766],
#        [ 20.96774194,  95.45454545],
#        [ 34.19354839, 110.60606061],
#         [ 16.12903226, 118.83116883],])

# For rooftop
# hull = np.array([[-18.70967742,  93.07359307],
#        [-34.19354839,  84.41558442],
#        [ -0.21505376,  16.45021645],
#        [ 14.83870968,  23.80952381],
#         [-18.70967742,  93.07359307],])
# for remote
hull = np.array([[-10.96774194,  58.44155844],
       [ -3.65591398,  34.1991342 ],
       [-10.53763441,  30.3030303 ],
       [ -5.37634409,  24.24242424],
       [ 15.69892473,  17.31601732],
       [ 29.03225806,  34.63203463],
       [ 18.70967742,  55.41125541],
        [-10.96774194,  58.44155844]])

in_zone = np.array([[-16.77419355,  24.02597403],
       [-13.22580645,  32.68398268],
       [  2.58064516,  26.62337662],
       [ -0.32258065,  16.23376623]])

out_zone = np.array([[ 1.48387097e+01,  1.25324675e+02],
       [-1.42108547e-14,  9.02597403e+01],
       [ 2.83870968e+01,  8.07359307e+01],
       [ 4.19354839e+01,  1.15367965e+02]])





p_arr_all = np.empty((0, 5))
rd = 0
cd = 0
d_weight = np.array([1.25998060e-01, 0.00000000e+00, 3.72307987e-04, 2.97648267e-04,
                     4.07682852e-03, 3.03747762e-03, 2.07011840e-02, 2.03772601e-01,
                     5.00000000e-01])

path = mpltPath.Path(hull)
zone_2 = mpltPath.Path(in_zone)
zone_1 = mpltPath.Path(out_zone)
i = next(bg)

c_list = np.empty((0,4))
pts_list = []
broken_radar_track = {'case_1': [], 'case_2': [], 'case_3': [], 'case_4': []}
start_list = [-1] * 8000
end_list = [-1] * 8000
prev_track_list = [-1] * 8000
unconfirmed_track = 0
road_objects_c = []
road_objects_r = []
road_objects_dict = dict()
road_objects_dict['no_match'] = [[], [], 0]
radar_id_count = 0
radar_ids = [-1]
def animate(g):
    global image_np, g2c_p, r2c_e, c2g
    global p_arr_all
    global camera_time_0
    global update
    global cam1
    global tracked_object_radar
    global alive_track
    global life
    global trackList
    global lastTrackIdx
    global k, G, H, Q, R, N
    global t_array
    global dt_array
    global t_no_matching
    global dt_no_match_array
    global p_time
    global radar_p_time
    global newcameramtx
    global dist
    global mtx
    global roi
    global cam_msg
    global model
    global idx
    global epoch
    global car_count
    global bus_count
    global person_count
    global truck_count
    global hull
    global run_cam_d
    global rd
    global cd
    global outputs
    global d_weight
    global c_list, pts_list
    global broken_radar_track, unconfirmed_track
    global path, zone_2, zone_1
    global start_list, prev_track_list, road_objects_c, road_objects_r, road_objects_dict, radar_id_count, radar_ids
    global arr_all, pc, arr_concat, p_arr_all
    if idx <= 0:
        i = next(bg)
        # read ros Topic camera or radar
        idx+=1
    else:
        i = next(bg)
        sensor = frame.load_data(i)
        if sensor == "/Radar":
            npts = frame.radar.message.width
            arr_all = pc2_numpy(frame.radar.message, npts)
            # arr_concat = np.vstack((arr_all, p_arr_all))
            arr_concat = arr_all
            p_arr_all = arr_concat.copy()
        frame_check = frame.full_data
        if frame_check:

            # print(abs(abs(rd- frame.radar.message.header.stamp.to_sec())))
            # rd = frame.radar.message.header.stamp.to_sec()
            # print(abs(abs(cd - frame.camera.message.header.stamp.to_sec())))
            # cd = frame.camera.message.header.stamp.to_sec()
            # print(frame.camera.message.header.stamp.to_sec() - frame.radar.message.header.stamp.to_sec())
            # print(frame.camera.message.header.stamp.to_sec() - frame.radar.message.header.stamp.to_sec())
            # print(frame.radar.message.header.stamp.to_sec())
            # print(frame.radar.message.header.stamp.to_sec()- epoch)
            # epoch = frame.radar.message.header.stamp.to_sec()
            arr_c = transform_radar(arr_concat.T, r2c_e).T  # from radar to camera (not pixel)
            arr_g = transform_radar(arr_c.T, c2g).T  # from camera to global
            mask = path.contains_points(np.vstack((arr_g[:, 0], arr_g[:, 2])).T)

            arr = arr_g[mask]
            pc = arr[:, :4]
            image_np = imgmsg_to_cv2(frame.camera.message)
            plt.cla()
            axs.set_xlim(-100, 100)
            axs.set_ylim(-100, 100)
            plt.plot(hull[:, 0], hull[:, 1], 'k-')

            # draw points on plt figure
            # axs.scatter(arr_g[:, 0], arr_g[:, 1], s=0.5)
            # arr = filter_zero(arr_g)
            #
            # draw points on plt figure

            # Perform class specific DBSCAN
            total_box, cls = dbscan_cluster(pc, eps=2.5, min_sample=15, axs=axs)
            total_box_1, cls_1 = dbscan_cluster(pc, eps=2, min_sample=2, axs=axs)

            if isinstance(cls, type(None)):
                cls = []
            if isinstance(cls_1, type(None)):
                cls_1 = []
            total_box = np.vstack((total_box, total_box_1))
            # print(total_box)
            box_index = non_max_suppression_fast(total_box[:, :4], .2)
            cls.extend(cls_1)
            cls = [cls[ii] for ii in box_index]
            total_box = total_box[box_index, :]
            # KF tracking
            detection_list = []

            if run_cam_d:
                _, _, C_M, camera_detection, outputs = process_track(image_np, i, curr_frames, prev_frames, outputs,
                                                     device, model, stride, names, pt, imgsz, cfg, strongsort_list, dt,
                                                     seen, half, classes=[2], conf_thres=0.75)
                # [5, 7, 0, 2]
                im0 = C_M.copy()
            if cls:
                features = np.empty((0, 12))
                radar_detection = []
                measSet = np.empty((0, 4))
                for ii, cc in enumerate(cls):
                    plot_box(total_box[ii, :], cc, axs)
                    centroid = np.mean(cc, axis=0)
                    # get tracking measurement
                    measSet = np.vstack((measSet, centroid))
                    bbox = get_bbox_cls(cc)
                    features = np.vstack((features, np.array(get_features(cc, bbox))))
                    bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)
                    bbox = project_to_image(bbox, g2c_p)
                    pts = project_to_image(cc.T, g2c_p)
                    box2d = get_bbox_2d(pts.T)
                    cv2.rectangle(im0, box2d[0], box2d[1], (255, 255, 0))
                    box2d = [box2d[0][0], box2d[0][1], box2d[1][0], box2d[1][1]]
                    box2d = convert_topy_bottomx(box2d)
                    radar_detection.append([cc, centroid, bbox, box2d])
                # measSet = measSet[:, :2].T
                measSet = np.vstack((measSet[:, 0].T, measSet[:, 2].T))
                features = scaler_model.fit_transform(features)
                prediction = svm_model.predict_proba(features)
                for ii in range(len(cls)):
                    pd = np.argmax(prediction[ii, :])
                    radar_detection[ii].append(prediction[ii, :])
                radar_2d = np.asarray([ii[3] for ii in radar_detection])
                cam_2d = np.asarray([ii[0] for ii in camera_detection])

                if cam_2d.any():
                    # print(cam_2d[0])
                    # print(convert_topy_bottomx([cam_2d[0][0], cam_2d[0][1], cam_2d[0][2], cam_2d[0][3]]))
                    # for debugging
                    # dis = image_np.copy()
                    # for ii in cam_2d:
                    #     cv2.rectangle(dis, (int(ii[0]), int(ii[1])), (int(ii[2]), int(ii[3])), (255, 255, 0))
                    # for ii in radar_2d:
                    #     cv2.rectangle(dis, (ii[0], ii[1]), (ii[2], ii[3]), (255, 255, 255))
                    # cv2.imshow('dis', dis)
                    # cv2.waitKey(100)

                    radar_matched, camera_matched, ious, radar_unmatched, camera_unmatched = match_detection(radar_2d, cam_2d)
                    for ii in range(len(radar_matched)):
                        display = image_np.copy()
                        detection_list.append(DetectedObject(r_d=radar_detection[radar_matched[ii]], c_d=camera_detection[camera_matched[ii]]))
                        # for debugging
                        # cv2.rectangle(display, (radar_detection[radar_matched[ii]][3][0], radar_detection[radar_matched[ii]][3][1]),
                        #               (radar_detection[radar_matched[ii]][3][2], radar_detection[radar_matched[ii]][3][3]), (255, 255, 0))
                        #
                        # cv2.rectangle(display, (int(camera_detection[camera_matched[ii]][0][0]), int(camera_detection[camera_matched[ii]][0][1])),
                        #               (int(camera_detection[camera_matched[ii]][0][2]), int(camera_detection[camera_matched[ii]][0][3])), (255, 0, 0))
                        # cv2.imshow('0', display)
                        # cv2.waitKey(100)
                    for ii in radar_unmatched:
                        detection_list.append(DetectedObject(r_d=radar_detection[ii]))
                    for ii in camera_unmatched:
                        detection_list.append(DetectedObject(c_d=camera_detection[ii]))
                else:
                    for ii in radar_detection:
                        detection_list.append(DetectedObject(r_d=ii))


            # if no radar detection and only camera detection
            elif camera_detection:
                # initiate empty radar measurement if no object is detected on radar
                # initiate all detections for camera
                measSet = np.empty((2, 0))
                for ii in camera_detection:
                    detection_list.append(DetectedObject(c_d=ii))
            else:
                # initiate measurement if no object is detected in both camera and radar
                measSet = np.empty((2, 0))
            # perform tracking
            trackList, unassignedMeas = gating(trackList, lastTrackIdx, PG, MP_IMM, maxVals, sensorPos, measSet, k)
            # perform gating
            trackList = updateStateTracks(trackList, lastTrackIdx, filterType, measSet, maxVals,
                                          lambdaVal, MP_IPDA, PG, PD, sensorPos, SVSFParams, k)
            # update the state of each track
            # trackList = updateTracksStatus(trackList,lastTrackIdx, delTenThr, delConfThr, confTenThr,k)

            if useLogic == True:
                trackList = updateTracksStatus_MN(trackList, lastTrackIdx, N1, M2, N2, N3, k)
            else:
                trackList = updateTracksStatus(trackList, lastTrackIdx, delTenThr, delConfThr, confTenThr,
                                               k)  # update the status of each track usiing the track manager

            # update the status of each track usiing the track manager
            # initiate tracks for measurements that were not gated or in other words unassigned measurements

            if isMM == True:
                trackList, lastTrackIdx = initiateTracksMM(trackList, lastTrackIdx, unassignedMeas, maxVals, G_List, H,
                                                           Q_List, R, models, filters, Ts, pInit, k, sensorType,
                                                           sensorPos, N)
            else:
                trackList, lastTrackIdx = initiateTracks(trackList, lastTrackIdx, measSet, maxVel, maxAcc, omegaMax, G,
                                                         H, Q, R, modelType, Ts, pInit, k, sensorType, sensorPos, N)

            # iterate through all tracks
            # ToDo change tracklet format so it doesn't need to iterate through all previous tracks
            # print('_____________________________________')
            # print(f'{k=}')
            camera_ids = [ii[2] for ii in camera_detection]
            # print(detection_list[0].cam_id)
            for ii in camera_ids:
                if ii not in road_objects_dict.keys():
                    # index 0 is for counting radarID, index 1 is for storing index in trackList
                    road_objects_dict[ii] = [[], [], 0]
                else:
                    road_objects_dict[ii][2] += 1
            matched_cameras = []
            for jj, ii in enumerate(trackList[:lastTrackIdx]):
                if lastTrackIdx == -1:
                    break
                # print(ii.BLs)
                # get centroid
                # check if track started in correct position
                centroid = ii.xPost[:2]

                track_label = None
                # if not ii.endSample and ii.status == 2:
                # if jj in [4, 16, 29, 33]:

                if not ii.endSample:
                    if jj == 28:
                        axs.text(centroid[0], centroid[1] - 2, 'ID: ' + str(jj), fontsize=11,
                                 color='r')
                    # axs.scatter(centroid[0], centroid[1], s=5, color='r')
                    if start_list[jj] == -1:
                        ii.ID = max(radar_ids) + 1
                        radar_ids.append(ii.ID)
                        radar_id_count += 1
                        if zone_1.contains_points(centroid.reshape(1, -1)):
                            zone = 1
                        elif zone_2.contains_points(centroid.reshape(1, -1)):
                            zone = 2
                        else:
                            zone = 0
                        start_list[jj] = [k, zone]
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
                    centroid_img[:, 2] = centroid[0]
                    centroid_img[:, 0] = centroid[1]
                    centroid_img[:, 1] = 0
                    pts = project_to_image(centroid_img.T, g2c_p).flatten()
                    s_img = np.zeros((1, 4))
                    s_img[:, :2] = [x1, y1]
                    s_img[:, 2] = -2
                    s_pts = project_to_image(s_img.T, g2c_p).flatten()
                    matched_measurement = match_measurement(detection_list, centroid)
                    # if jj == 25:
                    #     print(ii.c_ID)
                    if matched_measurement != None:
                        # if at the detection stage a camera was matched with the radar
                        if detection_list[matched_measurement].cam_id:
                            ii.c_ID.append(detection_list[matched_measurement].cam_id)

                        tracked_list[jj].dets.append(detection_list[matched_measurement])
                        tracked_list[jj].start = k
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
                        trk = DetectedObject(trk=[jj, centroid, c_track_id])
                        detection_list.append(trk)
                        # If this radar ID isn't matched with the camera previously add it
                    # axs.plot([centroid[0], x1], [centroid[1], y1], linewidth=2, markersize=5)
                    c_sx = s_pts[0] - pts[0]
                    c_sy = s_pts[1] - pts[1]
                    angle = math.atan2(c_sy, c_sx)
                    c_x1 = math.cos(angle) * 40 + pts[0]
                    c_y1 = math.sin(angle) * 40 + pts[1]
                    # cv2.line(im0, (int(pts[0]), int(pts[1])), (int(c_x1), int(c_y1)), color=(0, 255, 0), thickness=2)
                    if tracked_list[jj].dets:
                        if run_cam_d:
                            radar_label, centroid_track, num_pts, cam_label, cam_box, cam_id = tracked_list[jj].get_prediction(
                                camera=run_cam_d)
                        else:
                            radar_label, num_pts, centroid_track = tracked_list[jj].get_prediction(
                                camera=run_cam_d)
                        track_label = distance_weighted_voting_custom(radar_label, centroid_track, d_weight)
                        ii.label = track_label

                elif ii.endSample:
                    # gather info about the track including camera ID
                    if prev_track_list[jj] == -1:
                        if run_cam_d:
                            radar_label, r_centroid, num_pts, cam_label, cam_box, cam_id = tracked_list[jj].get_prediction(
                                camera=run_cam_d)
                        else:
                            radar_label, r_centroid, num_pts = tracked_list[jj].get_prediction(
                                camera=run_cam_d)
                        track_label = distance_weighted_voting_custom(radar_label, r_centroid, d_weight)
                    else:
                        unconfirmed_track +=1
                        # axs.plot(centroid[0], centroid[1], marker="o", markersize=5)
                        # print(jj)
                        # print(tracked_list[jj].start)
                    # tracked_object[jj] = None
                    alive_track[jj] = False
                    continue
                # add if first appearance
                if jj not in tracked_object_radar.keys():
                    tracked_object_radar[jj] = radar_object((centroid[0], centroid[1]))
                    alive_track[jj] = True
                # update if in tracked objects
                elif jj in tracked_object_radar.keys() and tracked_object_radar[jj]:
                    tracked_object_radar[jj].upd((centroid[0], centroid[1]))
                        # axs.text(centroid[0], centroid[1] - 5, 'Speed: ' + f'{speed*3.6:.2f} km/h', fontsize=11,
                        #          color='r')
                        # axs.text(centroid[0], centroid[1] - 12, 'Prob: ' + f'{pCurrent:.2f}', fontsize=11,
                        #          color='r')

                        # if latency >=0:
                        #    axs.text(centroid[0], centroid[1] - 8, 'Latency: ' + f'{latency*Ts:.2f} s', fontsize=11,
                        #             color='r')

                # plot and update all alive tracks
                # for tk, j in enumerate(alive_track[:lastTrackIdx]):
                #     if j:
                #         tracked_object[tk].life -= 1
                #         x = [obtk[0] for obtk in tracked_object[tk].tracklet]
                #         y = [obtk[1] for obtk in tracked_object[tk].tracklet]
                #         axs.plot(x, y, mew=0)
                #         if tracked_object[tk].life == 0:
                #             alive_track.remove(tk)
            # print(road_objects_c)
            # print(road_objects_r)
            for ii in detection_list:
                # check if this camera id then radar id is in any of the tracks
                in_track = False
                ids = []
                if ii.rad_id == 28:
                    print(ii.cam_id)
                for jj, tk in enumerate(tracked_object_all):
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
                # if ii.cam_id in [5]:
                #     print(ii.cam_id)
                #     print(ii.rad_id)
                #     print('_____')
                if in_track:
                    # merge all the track into one track
                    # keep_track = ids[0]
                    # if len(ids) > 1:
                    #     for jj in ids[1:]:
                    #         for jjj in tracked_object_all[jj].c_id:
                    #             if jjj not in tracked_object_all[keep_track].c_id:
                    #                 tracked_object_all[keep_track].c_id.insert(0, jjj)
                    #         for jjj in tracked_object_all[jj].r_id:
                    #             if jjj not in tracked_object_all[keep_track].r_id:
                    #                 tracked_object_all[keep_track].r_id.insert(0, jjj)
                    #         tracked_object_all[jj].delete()
                    # if camera id is found
                    tracked_object_all[ids[0]].activate = ii.sensor
                    # add the radar id to this tracked object if not there
                elif ii.cam_id or ii.rad_id:
                    trk = TrackedObjectALL(c_id=ii.cam_id, r_id=ii.rad_id, sensor=ii.sensor)
                    tracked_object_all.append(trk)

            #
            # print(len(camera_ids))
            # print(len(tracked_object_all))
            # print(camera_detection)
            for ii in detection_list:
                if ii.sensor in ['Both', 'Camera']:
                    box = ii.cam_box
                    py = min([box[1], box[3]])
                    px = (box[0]+box[2])/2
                    g_x, g_z = Cam2Ground(px, py, c2wcs)
                    axs.scatter(-g_z, g_x, s=5)
                    # print([g_x, g_z])
                    # print(ii.centroid)

            # for ii in camera_detection:
            #     if ii[2] == 9:
            #         # print(ii)
            #         cv2.rectangle(image_np, (int(ii[0][0]), int(ii[0][1])), (int(ii[0][2]), int(ii[0][3])), color=(255,255,255))
            #         cv2.imshow('0', image_np)
            #         cv2.waitKey(0)
            # print(outputs[0][:, 4])
            font_size = 0.3
            thickness = 1
            # if np.asarray(outputs[0]).any():
            #     print(outputs[0][:, 4])
            for ii, trk in enumerate(tracked_object_all):
                if True:
                    if not trk.deleted and trk.activate:
                        if trk.activate in ['Radar', 'Both', 'Radar_track'] and trk.r_id:
                            r_id = trk.r_id
                            r_trk = trackList[r_id[-1]]
                            centroid = r_trk.xPost[:2]
                            speed = r_trk.speed
                            centroid_img = np.zeros((1, 4))
                            centroid_img[:, 0] = centroid[0]
                            centroid_img[:, 2] = centroid[1]
                            centroid_img[:, 1] = 0
                            pts = project_to_image(centroid_img.T, g2c_p).flatten()
                            cv2.putText(im0, f'Fusion ID: {ii}', (
                                int(pts[0] + 10), int(pts[1]) - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        font_size, (0, 255, 0), thickness, cv2.LINE_AA)
                            cv2.putText(im0, f'Radar_id: {r_id}', (
                                int(pts[0] + 10), int(pts[1]) +15),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        font_size, (0, 255, 0), thickness, cv2.LINE_AA)
                            cv2.putText(im0, f'Speed: {speed:.03g}km/h', (
                                int(pts[0]) + 10, int(pts[1]) + 3),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        font_size, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.circle(im0, (int(pts[0]), int(pts[1])), 5, color=(0, 255, 0), thickness=2)
                            if trk.c_id:
                                cv2.putText(im0, f'camera_id: {trk.c_id}', (
                                    int(pts[0] + 10), int(pts[1]) + 25),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            font_size, (0, 255, 0), thickness, cv2.LINE_AA)
                            trk.life = 10
                        elif trk.c_id:
                            c_id = trk.c_id
                            if c_id[-1] in outputs[0][:, 4]:
                                id= np.where(outputs[0][:, 4] == c_id[-1])[0][0] # what?
                                bbox = outputs[0][id, :4]
                                pts = [(bbox[2] - bbox[0]) / 2 + bbox[0], bbox[3] - 10]
                                if not trk.r_id:
                                    cv2.putText(im0, f'Fusion ID: {ii}', (
                                        int(pts[0] + 10), int(pts[1])),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                font_size, (0, 0, 255), thickness, cv2.LINE_AA)
                                    cv2.circle(im0, (int(pts[0]), int(pts[1])), 5, color=(0, 0, 255), thickness=2)
                                    cv2.putText(im0, f'camera_id: {c_id}', (
                                        int(pts[0] + 10), int(pts[1])+ 13),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                font_size, (0, 0, 255), thickness, cv2.LINE_AA)
                                if trk.r_id:
                                    r_id = trk.r_id
                                    speed = trackList[r_id[-1]].speed
                                    cv2.circle(im0, (int(pts[0]), int(pts[1])), 5, color=(0, 0, 255), thickness=2)
                                    cv2.putText(im0, f'Fusion ID: {ii}', (
                                        int(pts[0] + 10), int(pts[1])- 15),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                font_size, (0, 255, 0), thickness, cv2.LINE_AA)
                                    cv2.putText(im0, f'camera_id: {c_id}', (
                                        int(pts[0] + 10), int(pts[1])+25),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                font_size, (0, 0, 255), thickness, cv2.LINE_AA)
                                    cv2.putText(im0, f'Speed: {speed:.03g}km/h', (
                                        int(pts[0]) + 10, int(pts[1]) + 3),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                font_size, (0, 0, 255), thickness, cv2.LINE_AA)
                                    cv2.putText(im0, f'Radar_id: {r_id} (noD)', (
                                        int(pts[0] + 10), int(pts[1]) + 15),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                font_size, (0, 0, 255), thickness, cv2.LINE_AA)
                                    trk.life = 10
                        trk.activate = False

                # elif not trk.deleted:
                #     trk.life -= 1
                #     if trk.life == 0:
                #         trk.delete()





            # print(outputs)
            # for jj, ii in enumerate(camera_ids):
            #     if ii not in matched_cameras:
            #         if road_objects_dict[ii][0]:
            #
            #             radar_id = road_objects_dict[ii][0][-1]
            #             track_index = road_objects_dict[ii][1][-1]
            #             speed = trackList[track_index].speed
            #             bbox = camera_detection[jj][0]
            #             pts = [(bbox[2]-bbox[0])/2 + bbox[0], bbox[3]-10]
            #             cv2.putText(im0, f'Radar_id: {radar_id} (noD)', (
            #                 int(pts[0] + 10), int(pts[1]) - 10),
            #                         cv2.FONT_HERSHEY_SIMPLEX,
            #                         0.5, (0, 0, 255), 1, cv2.LINE_AA)
            #             cv2.putText(im0, f'camera_id: {ii}', (
            #                 int(pts[0] + 10), int(pts[1]) + 20),
            #                         cv2.FONT_HERSHEY_SIMPLEX,
            #                         0.5, (0, 0, 255), 1, cv2.LINE_AA)
            #             cv2.putText(im0, f'Speed: {speed:.03g}km/h', (
            #                 int(pts[0]) + 10, int(pts[1]) + 3),
            #                         cv2.FONT_HERSHEY_SIMPLEX,
            #                         0.5, (0, 0, 255), 1, cv2.LINE_AA)
            #             cv2.circle(im0, (int(pts[0]), int(pts[1])), 5, color=(0, 0, 255), thickness=2)
            #         else:
            #             bbox = camera_detection[jj][0]
            #             pts = [(bbox[2] - bbox[0]) / 2 + bbox[0], bbox[3] - 10]
            #             cv2.putText(im0, f'camera_id: {ii}', (
            #                 int(pts[0] + 10), int(pts[1]) + 20),
            #                         cv2.FONT_HERSHEY_SIMPLEX,
            #                         0.5, (0, 0, 255), 1, cv2.LINE_AA)
            #             cv2.putText(im0, f'Radar_id: No Radar', (
            #                 int(pts[0] + 10), int(pts[1]) - 10),
            #                         cv2.FONT_HERSHEY_SIMPLEX,
            #                         0.5, (0, 0, 255), 1, cv2.LINE_AA)
            #             cv2.putText(im0, f'Speed: No Radar', (
            #                 int(pts[0] + 10), int(pts[1]) + 3),
            #                         cv2.FONT_HERSHEY_SIMPLEX,
            #                         0.5, (0, 0, 255), 1, cv2.LINE_AA)
            #             cv2.circle(im0, (int(pts[0]), int(pts[1])), 5, color=(0, 0, 255), thickness=2)

            # for ii, jj in enumerate(road_objects_dict['no_match'][1]):
            #     print(jj)
            # print(f"dt = {i.message.header.stamp.to_sec() - cam_msg}")
            # yolo detection
            # cam1, detection = detect(source=cam1, model=model, device=device, colors=colors, names=names,
            #                              view_img=False)
            # Radar projection onto camera parameters

            # axs.text(0, 60, f'Car_Count {car_count}', fontsize=11, color='b')
            # axs.text(0, 70, f'Bus_Count {bus_count}', fontsize=11, color='b')
            # axs.text(0, 80, f'Person_Count {person_count}', fontsize=11, color='b')
            # axs.text(0, 90, f'Truck_Count {truck_count}', fontsize=11, color='b')
            # for c in cluster_hist[0]:
            #     # print(c)
            #     axs.scatter(c[0][:, 0], c[0][:, 1], s=0.5)
            # input()
            # pickle.dump(c_list, open('c_list.pkl', 'wb'))
            # pickle.dump(pts_list, open('pts_list.pkl', 'wb'))
            # print(car_count)

            im0, cam_arr = render_radar_on_image(arr_g, im0, g2c_p, 9000, 9000)
            cv2.imshow('Camera', im0)
            # cv2.imshow('Camera', image_np)
            # cv2.imshow('Camera', C_M)
            cv2.waitKey(1)
            k += 1
            update = 1

            # print(time.time() - ts)
            car_count = 0
            for ii in road_objects_dict.keys():
                if road_objects_dict[ii][2] > 5:
                    car_count += 1
            idx += 1
            p_arr_all = arr_all.copy()

            # print(car_count)



ani = FuncAnimation(fig, animate, interval=10, frames=2000)
plt.show()


""""                
                    # cc = det.cls
                    # bbox = get_bbox_cls(cc)
                    # det.centroid = np.mean(det.cls, axis=0)
                    # det.rad_box = bbox
                    # features = np.vstack((features, np.array(get_features(cc, bbox))))
                    # axs.text(bbox[0], bbox[1] - 2, 'SVM: ' + str(classes[int(prediction[0])]), fontsize=11,
                    #          color='r')
                    # print(bbox)
                    # bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)
                    # bbox = project_to_image(bbox, r2c)
                    # pts = project_to_image(cc.T, r2c)
                    # box2d = get_bbox_2d(pts.T)
                    # cv2.rectangle(cam1, box2d[0], box2d[1], (255, 255, 0))
                    # box2d = [box2d[0][0], box2d[0][1], box2d[1][0], box2d[1][1]]
                    # box2d = convert_topy_bottomx(box2d)
                    # det.rad_box_cam_coord = box2d
                    # if camera_detection:
                    #     matched = find_gt(box2d, detection)
                    #     det.cam_label = classes.index(matched[0][1])
                    #     det.cam_box = matched[0][0]
                    #     det.cam_rad_iou = matched[1]

                    # cv2.putText(cam1, f'SVM: {classes[int(prediction[0])]}', box2d[0],
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             1, (255, 255, 255), 2, cv2.LINE_AA)

                    # draw_projected_box3d(cam1, bbox)

                # print(features)
                # prediction = svm_model.predict_proba(features)
                for ii, det in enumerate(detection_list):
                    pd = np.argmax(prediction[ii, :])
                    box2d = det.rad_box_cam_coord
                    det.rad_label = prediction[ii, :]
                    # axs.text(bbox[0], bbox[1] - 2, 'SVM: ' + str(classes[int(pd)]), fontsize=11,
                    #          color='r')
                    cv2.putText(image_np, f'SVM: {classes[int(pd)]}', (box2d[0], box2d[1]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 1, cv2.LINE_AA)
                # print(prediction)
                # convert clusters into tracking input format
                for cc in cls:
                    measSet = np.vstack((measSet, np.mean(cc, axis=0)))
                measSet = measSet.T[:2, :]"""



"""
Key issue

Uneven sampling time

Track object where each frame might have multiple detections
Track probability not is not decreasing
Reason: clutters?, th
Add Constant acceleration model
Implement IMM for multi target

If there is a object ID that is > the len of the object list
    Append object
    
    
check all alive track
if no update -1 to life
draw all track with life > 0
Radar output:
    xyz
    radar power
    range_rate (doppler): radial velocity 
    
DBSCAN Detection output
    Centroid: center of the bounding box
    Bounding box width length height (orientation)
    average range_rate: average of xyz of the clusters
    



Tracking 
    We try r, theta, v_r directly with EKF
    Data association -> PDA
    Filter -> SVSF
        Input: x y
        Input: r, theta, v_r
        Measurement covariance
        Previous cov
        previous mode
        --> IMM-SVSF-EKF
            IMM three mode
                constant velocity
                constant acceleration
                constant turn
        Output
        cov, state
        
        
Bell camera

"""