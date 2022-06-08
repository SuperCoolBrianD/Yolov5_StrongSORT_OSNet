import sys

import numpy as np

sys.path.append('yolor')
sys.path.append('SVSF_Track')
# from yolor.detect_custom import init_yoloR, detect
from SVSF_Track.MTT_Functions import *
from radar_utils import *
# import sort
import rosbag
from matplotlib.animation import FuncAnimation

# Read recording
bag = rosbag.Bag("record/traffic3.bag")
# bag = rosbag.Bag("record/traffic3.bag")
# bag = rosbag.Bag("record/traffic1.bag")
topics = bag.get_type_and_topic_info()

# old SORT tracker
# mot_tracker = sort.Sort(min_hits=2, max_age=8, iou_threshold=0.1)

for i in topics[1]:
    print(i)

radar_d = '/radar_data'
# init plt figure
fig, axs = plt.subplots(1, figsize=(6, 6))
fig.canvas.set_window_title('Radar Detection and Tracking IMM')
# create generator object for recording
bg = bag.read_messages()
s = 0
# model, device, colors, names = init_yoloR(weights='yolor/yolor_p6.pt', cfg='yolor/cfg/yolor_p6.cfg',
#                                           names='yolor/data/coco.names', out='inference/output', imgsz=640)
# adjust image visualization
cv2.imshow('Camera', np.zeros((480, 640)))
cv2.moveWindow('Camera', 800, 800)
image_np = np.empty((5, 5))
# init loop
framei = 0
imgs = []
radar_time_0 = 0
camera_time_0 = 0
update = 1
cam1 = np.empty((5,5))
tracked_object = dict()
alive_track = [0] * 8000
life = 10
# track parameters
trackList = [0] * 8000  # allocate track list, a list of objects
lastTrackIdx = -1  # index of a track in the list,
# init KF tracker


Ts = .12 #sampling time
unassignedMeas0 = np.array([[]])

modelType = 'CV' #model used for single model filters
sensor = 'Lidar'
# filterType = "IPDAGVBLSVSF"
filterType = "IMMIPDAKF"
# filterType = "IPDAGVBLSVSF"
# filterType = "IPDASVSF"
sensorPos = np.array([0, 0])

# new for svsf

n = 7 # x, y, vx, vy, ax, ay, turn-rate
m = 2 #number of measurements
psi1 = 100/2 # p larger uncertainty increase
psi2 = 100/2 # v larger uncertainty increase
psi3 = 10 # a larger uncertainty increase
psi4 = 10 # turn rate larger uncertainty increase
gammaZ = .1 * np.eye(m) # convergence rate stability from 0-1 for measured state
gammaY = .1 * np.eye(n - m) # for unmeasured state

psiZ = np.array([psi1, psi1])
psiY = np.array([psi2, psi2, psi3, psi3, psi4])
T_mat = np.eye(n)

SVSFParams = [psiZ,psiY,gammaZ,gammaY,T_mat]

#Standard deviations for process and measurement noises
sigma_v = 1E-1 #process noise standard deviation
sigma_v_filt = sigma_v #process noise standard deviation for filter
sigma_w = .5 #measurement noise standard deviation in position

# Process noise covariances
Q_CV = np.diag([sigma_v**2,sigma_v**2,0]) #process noise co-variance for CV model
Q_CA = np.diag([sigma_v**2,sigma_v**2,0]) #process noise co-variance for CA model
Q_CT = np.diag([sigma_v**2,sigma_v**2, sigma_v**2]) #process noise co-variance for CT model

R = np.diag(np.array([sigma_w ** 2, sigma_w ** 2]))  # measurement co-variance
H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]])  # measurement matrix

#Input gain matrices
G_CV = np.array([[(Ts**2)/2, 0,0],[0, (Ts**2)/2,0],[Ts,0,0],[0,Ts,0],[0,0,0],[0,0,0],[0,0,0]])  #input gain for CV
G_CA = np.array([[(Ts**2)/2, 0,0],[0, (Ts**2)/2,0],[Ts,0,0],[0,Ts,0], [1,0,0],[0,1,0],[0,0,0]]) #input gain for CA
G_CT = np.array([[(Ts**2)/2, 0,0],[0, (Ts**2)/2,0],[Ts,0,0],[0,Ts,0],[0,0,0],[0,0,0],[0,0,Ts]]) #input gain for CT

#Parameters for 1-point initialization
maxAcc = 5
maxVel = 16 #for initializing velocity variance for 1-point initialization
omegaMax = math.radians(4) #for initializing turn-rate variance for 1-point initialization
maxVals = [maxVel, maxAcc, omegaMax]

#Parameters for IPDA
pInit = .2 #initial probability of track existence
PD = .6 #probability of target detection in a time step
PG = .99999 #gate probability
lambdaVal = 0.05 # parameter for clutter density

delTenThr = .05 #threshold for deleting a tenative track
confTenThr = .6 # threshold for confirming a tenative track
delConfThr = 0.05 # threshold for deleting a confirmed track

#IMM parameters
models = ["CV", 'CT']
filters = ['IPDAKF', 'IPDAKF']
G_List = [G_CV, G_CT] #input gain list
Q_List = [Q_CV, Q_CT] #process noise co-variance list
#pInits = [.2,.2] #initial track existence probabilities
#uVec0 = [.5, .5] #initial mode probabilities
r = len(models)

#Set Markov matrix for IMM below
P_ii_IMM = .95
P_ij_IMM = (1-P_ii_IMM)/r
if r==2:
    MP_IMM = np.array([[P_ii_IMM, P_ij_IMM], [P_ij_IMM, P_ii_IMM]])
elif r==3:
    MP_IMM = np.array([[P_ii_IMM,P_ij_IMM,P_ij_IMM],[P_ij_IMM,P_ii_IMM,P_ij_IMM],[P_ij_IMM,P_ij_IMM,P_ii_IMM]])


P_ii = .95  #for Markov matrix of IPDA
MP_IPDA = np.array([[P_ii, 1-P_ii], [1-P_ii, P_ii]]) #Markov matrix for IPDA




##Best for CV
'''
gammaZ = 0.778*np.eye(m)
gammaY = 0.778*np.eye(n-m)
psi1 = 9333.0
psi2 = 40.0
psi3 = 10 #arbitrary 
'''



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


mtx = np.array([[1113.5, 0., 974.2446],
                [0.,1113.5,586.6797],
                [0., 0., 1.]])

# mtx = np.array([[234.45076996, 0., 334.1804498],
#                 [0.,311.6748573,241.50825294],
#                 [0., 0., 1.]])
dist = np.array([[-0.06624252, -0.00059409, -0.00183169,  0.0030411,   0.00063524]])

h,  w = 480, 640
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


def animate(g):
    global image_np
    global framei
    global imgs
    global camera_time_0
    global update
    global cam1
    global tracked_object
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
    i = next(bg)
    # read ros Topic camera or radar
    if i.topic == '/usb_cam/image_raw/compressed':
        np_arr = np.frombuffer(i.message.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (640, 480), 5)
        # image_np = cv2.remap(image_np, mapx, mapy, cv2.INTER_LINEAR)
        # # crop the image
        # x, y, w, h = roi
        # image_np = image_np[y:y + h, x:x + w]
        cam1 = image_np
        # t_array.append((i[2], image_np))
        # t_no_matching = i[2]
        for tk, j in enumerate(alive_track[:lastTrackIdx]):
            if j:
                # print(tracked_object[tk].tracklet)
                x = [obtk[0][0] for obtk in tracked_object[tk].tracklet]
                y = [obtk[0][1] for obtk in tracked_object[tk].tracklet]
                axs.plot(x, y, mew=0)
    elif i.topic == '/radar_data' or 'radar_data':
        # print(i[2])
        # print(i.message.time_stamp)
        # select closest image to radar frame
        # min_t = 999999999
        # dt_no_match = abs(i[2].to_sec() - t_no_matching.to_sec())
        # dt = None
        #
        # radartime = radar_p_time - i.message.time_stamp[0]/10**6
        # radar_p_time = i.message.time_stamp[0]/10**6
        # ros_time = p_time - i[2].to_sec()
        # p_time = i[2].to_sec()
        # print(f"ros_time: {ros_time}")
        # print(f"radar_time: {radartime}")
        # print(f"delta lag: {abs(ros_time)-abs(radartime)}")
        # print(f"____________________________")
        # for c_frame in t_array:
        #     if abs(c_frame[0].to_sec()-i[2].to_sec()) < min_t:
        #         cam1 = c_frame[1]
        #         dt = abs(c_frame[0].to_sec()-i[2].to_sec())
        # if dt:
        #     dt_array.append(dt)
        # dt_no_match_array.append(dt_no_match)
        # t_array = []
        # print(dt)
        plt.cla()
        # record time (Radar time)
        tm = i.message.time_stamp[0]/10**6
        axs.set_xlim(-50, 80)
        axs.set_ylim(0, 100)
        # convert SRS message to Numpy
        arr = convert_to_numpy(i.message.points)
        axs.scatter(arr[:, 0], arr[:, 1], s=0.5)
        # Filter zero doppler points
        arr = filter_zero(arr)
        # draw points on plt figure

        pc = arr[:, :4]
        ped_box = np.empty((0, 5))
        # Perform class specific DBSCAN
        total_box, cls = dbscan_cluster(pc, eps=1.5, min_sample=20, axs=axs)
        # Do pedestrian if required
        # ped_box, ped_cls = dbscan_cluster(pc, eps=2, min_sample=10)

        if total_box.any() and ped_box.any:
            total_box = np.vstack((total_box, ped_box))
        measSet = np.empty((0, 4))
        # KF tracking
        if cls:
            # print(len(cls))
            # convert clusters into tracking input format
            for ii in cls:
                measSet = np.vstack((measSet, np.mean(ii, axis=0)))
            measSet = measSet.T[:2, :]
            # perform tracking
            trackList, unassignedMeas = gating(trackList, lastTrackIdx, PG, MP_IMM, maxVals, sensorPos, measSet)
            # perform gating
            trackList = updateStateTracks(trackList,lastTrackIdx, filterType, measSet, maxVals,
                                          lambdaVal,MP_IPDA, PG, PD, sensorPos, SVSFParams, k)
            # update the state of each track
            trackList = updateTracksStatus(trackList,lastTrackIdx, delTenThr, delConfThr, confTenThr,k)
            # update the status of each track usiing the track manager
            # initiate tracks for measurements that were not gated or in other words unassigned measurements
            trackList, lastTrackIdx = initiateTracksMM(trackList,lastTrackIdx, unassignedMeas, maxVals, G_List, H,
                                                       Q_List, R, models,filters, Ts, pInit, k, sensor, N)

            #input("")
            k += 1
            # iterate through all tracks
            # ToDo change tracklet format so it doesn't need to iterate through all previous tracks
            for jj, ii in enumerate(trackList[:lastTrackIdx]):
                # print(ii.BLs)
                # get centroid

                centroid = ii.xPost[:2]
                speed = np.sqrt(ii.xPost[2]**2 + ii.xPost[3]**2)
                latency = ii.latency
                pCurrent = ii.pCurrent
                status = ii.status
                # if track has terminated remove from tracked objects
                if ii.endSample:
                    tracked_object[jj] = None
                    alive_track[jj] = False
                    continue
                # add if first appearance
                if jj not in tracked_object.keys():
                    tracked_object[jj] = radar_object(((centroid[0], centroid[1]), tm))
                    alive_track[jj] = True
                # update if in tracked objects
                elif jj in tracked_object.keys() and tracked_object[jj]:
                    tracked_object[jj].upd(((centroid[0], centroid[1]), tm))
                    axs.text(centroid[0], centroid[1] - 2, 'ID: ' + str(jj), fontsize=11,
                             color='r')
                    axs.text(centroid[0], centroid[1] - 5, 'ID: ' + f'{speed*3.6:.2f} km/h', fontsize=11,
                             color='r')
                    axs.text(centroid[0], centroid[1] - 12, 'Prob: ' + f'{pCurrent:.2f}', fontsize=11,
                             color='r')

                    #if latency >=0:
                     #   axs.text(centroid[0], centroid[1] - 8, 'Latency: ' + f'{latency*Ts:.2f} s', fontsize=11,
                      #           color='r')

            # plot and update all alive tracks
            for tk, j in enumerate(alive_track[:lastTrackIdx]):
                if j:
                    tracked_object[tk].life -= 1
                    x = [obtk[0][0] for obtk in tracked_object[tk].tracklet]
                    y = [obtk[0][1] for obtk in tracked_object[tk].tracklet]
                    axs.plot(x, y, mew=0)
                    if tracked_object[tk].life == 0:
                        alive_track.remove(tk)
        if cam1.any():
            # yolo detection
            # cam1, detection = detect(source=cam1, model=model, device=device, colors=colors, names=names,
            #                              view_img=False)
            # Radar projection onto camera parameters
            ry = 0
            rz = .02
            tx = 0.1
            ty = 0.01
            tz = 0.1
            rx = 1.62
            r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)
            if cls:
                for cc in cls:
                    bbox = get_bbox_cls(cc)
                    # print(bbox)
                    bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[5], bbox[3], bbox[4], 0)
                    bbox = project_to_image(bbox, r2c)
                    pts = project_to_image(cc.T, r2c)
                    # print(pts)
                    box2d = get_bbox_2d(pts.T)
                    cv2.rectangle(cam1, box2d[0], box2d[1], (255,255,0))
                    draw_projected_box3d(cam1, bbox)
            img, cam_arr = render_radar_on_image(arr, cam1, r2c, 9000, 9000)
            cv2.imshow('Camera', img)
            cv2.waitKey(1)
            update = 1
        # print(f'Max dt = {max(dt_array)}')
        # print(f'Min dt = {min(dt_array)}')
        # print(f'Average dt = {sum(dt_array)/len(dt_array)}')
        # print(f'Max dt_no_match = {max(dt_no_match_array)}')
        # print(f'Min dt_no_match = {min(dt_no_match_array)}')
        # print(f'Average dt_no_match = {sum(dt_no_match_array)/len(dt_no_match_array)}')


ani = FuncAnimation(fig, animate, interval=10, frames=2000)
plt.show()


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