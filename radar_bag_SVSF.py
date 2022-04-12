import numpy as np
import sys
sys.path.append('yolor')
sys.path.append('SVSF_Track')
from yolor.detect_custom import init_yoloR, detect
import sort
import rosbag

import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from radar_utils import *
import torch
from SVSF_Track.MTT_Functions import *


bag = rosbag.Bag("record/ped_cross.bag")
topics = bag.get_type_and_topic_info()

mot_tracker = sort.Sort(min_hits=2, max_age=8, iou_threshold=0.1)

for i in topics[1]:
    print(i)

radar_d = '/radar_data'
bg = bag.read_messages()
s = 0
# model, device, colors, names = init_yoloR(weights='yolor/yolor_p6.pt', cfg='yolor/cfg/yolor_p6.cfg',
#                                           names='yolor/data/coco.names', out='inference/output', imgsz=640)
cv2.imshow('Camera', np.zeros((480, 640)))
cv2.moveWindow('Camera', 800, 800)
image_np = np.empty((5, 5))
framei = 0
imgs = []
radar_time_0 = 0
camera_time_0 = 0
update = 1
cam1 = np.empty((5,5))
tracked_object = dict()
alive_track = []
life = 10
trackList = [0] * 8000  # allocate track list, a list of objects
lastTrackIdx = -1  # index of a track in the list,
# init KF tracker
sigma_v = 1E-3 #process noise standard deviation
sigma_v_filt = sigma_v #process noise standard deviation for filter
sigma_w = .1 #measurement noise standard deviation in position
R = np.diag(np.array([sigma_w ** 2, sigma_w ** 2]))  # measurement co-variance
H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])  # measurement matrix
pInit = .2 #initial probability of track existence
PG = .9979 #gate probability
Ts = .12 #sampling time
unassignedMeas0 = np.array([[]])
sigma_w = .1
maxVel = 27 #for initializing velocity variance for 1-point initialization
omegaMax = math.radians(4) #for initializing turn-rate variance for 1-point initialization
modelType = 'CV'
sensor = 'Lidar'
filterType = "IPDAKF"
sensorPos = np.array([0, 0])
P_ii = .99  #for Markov matrix of IPDA
PD = .8 #probability of target detection in a time step
MP = np.array([[P_ii,1-P_ii],[1-P_ii,P_ii]]) #Markov matrix for IPDA
lambdaVal = 1E-4 #parameter for clutter density
delTenThr = .05 #threshold for deleting a tenative track
confTenThr = .9 #threshold for confirming a tenative track
delConfThr = 0.005 #threshold for deleting a confirmed track

if modelType=="CV":
    Q = np.diag(np.array([sigma_v_filt**2,sigma_v_filt**2,0**2]))
    G = np.array([[(Ts**2)/2, 0,0],[0, (Ts**2)/2,0],[Ts,0,0],[0,Ts,0],[0,0,0]])
    ''''
    Q_CV = L1*np.array([[(Ts**3)/3, 0, (Ts**2)/2, 0, 0],
                     [0,(Ts**3)/3, 0, (Ts**2)/2, 0 ],
                     [(Ts**2)/2, 0, Ts, 0, 0],
                     [0, (Ts**2)/2, 0, Ts, 0],
                    [0, 0, 0, 0, 0]])
    #Q=Q_CV
    '''
elif modelType=="CT":
    Q = np.diag(np.array([sigma_v_filt**2,sigma_v_filt**2,sigma_v_filt**2]))
    G = np.array([[(Ts**2)/2, 0,0],[0, (Ts**2)/2,0],[Ts,0,0],[0,Ts,0],[0,0,Ts]])
    '''
    Q_CT = L1*np.array([[(Ts**3)/3, 0, (Ts**2)/2, 0, 0],
                     [0,(Ts**3)/3, 0, (Ts**2)/2, 0 ],
                     [(Ts**2)/2, 0, Ts, 0, 0],
                     [0, (Ts**2)/2, 0, Ts, 0],
                    [0, 0, 0, 0, (L2/L1)*Ts]])
    #Q=Q_CT
    '''
N = 8000
trackList,lastTrackIdx = initiateTracks(trackList, lastTrackIdx, unassignedMeas0, sigma_w, maxVel, omegaMax, G, H, Q, R,
                                        modelType, Ts, pInit, 0, sensor, N)
k = 0
for its, i in enumerate(bg):
    if its < 400:
        continue
    # print(its)
    if i.topic == '/usb_cam/image_raw/compressed':
        # if camera_time_0 == 0:
        #     camera_time_0 = i.message.header.stamp

        np_arr = np.frombuffer(i.message.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if update:
            cam1 = image_np
            update = 0
    # print(i.message.header)
        # imgs.append(image_np)
    elif i.topic == '/radar_data' or 'radar_data':
        fig, axs = plt.subplots(1, figsize=(6, 6))

        plt.cla()
        tm = i.message.time_stamp[0]/10**6
        xlim = axs.set_xlim(-50, 80)
        axs.set_ylim(0, 100)
        arr = convert_to_numpy(i.message.points)
        vel = arr[:, -1]
        arr = filter_zero(arr)
        axs.scatter(arr[:, 0], arr[:, 1], s=0.5)
        pc = arr[:, :4]
        # print(f"unique speed for entire scene {np.unique(arr[:, -2])}")
        # print('_________________________________________________________________-')
        ped_box = np.empty((0, 5))
        total_box, cls = dbscan_cluster(pc, eps=2, min_sample=20, axs=axs)
        measSet = np.empty((1, 4))
        for ii in cls:
            measSet = np.vstack((measSet, np.mean(ii, axis=0)))
        measSet = measSet.T[:2, :]
        print(measSet)
        trackList, unassignedMeas = gating(trackList, lastTrackIdx, PG, sensorPos, measSet)
        for ii in trackList:
            if ii == 0:
                break
            print(ii.status)
        # perform gating
        trackList = updateStateTracks(trackList,lastTrackIdx, filterType, measSet, lambdaVal, MP, PG, PD, sensorPos, k)
        # update the state of each track
        trackList = updateTracksStatus(trackList,lastTrackIdx, delTenThr, delConfThr, confTenThr, k)
        # update the status of each track usiing the track manager
        #initiate tracks for measurements that were not gated or in other words unassigned measurements
        trackList, lastTrackIdx = initiateTracks(trackList,lastTrackIdx,unassignedMeas, sigma_w, maxVel, omegaMax, G, H, Q, R, modelType, Ts, pInit, k, sensor, N)
        k += 1
        # for ii in trackList[:lastTrackIdx]:
        #     print(ii.xPost)
        #     print(ii.pCurrent)
        # print(trackList)



        # if total_box.any() and ped_box.any:
        #     total_box = np.vstack((total_box, ped_box))
        # track_bbs_ids = mot_tracker.update(total_box)
        # # tracking
        # for t in range(track_bbs_ids.shape[0]):
        #     centroid = track_bbs_ids[t, :4]
        #
        #     if track_bbs_ids[t, -1] not in tracked_object.keys():
        #         alive_track.append(track_bbs_ids[t, -1])
        #         tracked_object[int(track_bbs_ids[t, -1])] = radar_object(((centroid[0], centroid[1]), tm))
        #     else:
        #         tracked_object[int(track_bbs_ids[t, -1])].upd(((centroid[0], centroid[1]), tm))
        #         tracked_object[int(track_bbs_ids[t, -1])].update_speed()
        #
        #     axs.text(track_bbs_ids[t, 0], track_bbs_ids[t, 1] - 2, 'ID: '+str(int(track_bbs_ids[t, -1])), fontsize=11,
        #              color='r')
        #     if tracked_object[int(track_bbs_ids[t, -1])].speed == 'Measuring':
        #         axs.text(track_bbs_ids[t, 0], track_bbs_ids[t, 1] - 5,
        #                  f'Speed: Measuring', fontsize=11,
        #                  color='r')
        #     else:
        #         axs.text(track_bbs_ids[t, 0], track_bbs_ids[t, 1] - 5, f'Speed: {float(tracked_object[int(track_bbs_ids[t, -1])].speed):.2f} km/h' , fontsize=11,
        #              color='r')
        #
        # for tk in alive_track:
        #     tracked_object[tk].life -= 1
        #     x = [obtk[0][0] for obtk in tracked_object[tk].tracklet]
        #     y = [obtk[0][1] for obtk in tracked_object[tk].tracklet]
        #     axs.plot(x, y, mew=0)
        #     if tracked_object[tk].life == 0:
        #         alive_track.remove(tk)
        # # print(alive_track)
        # # print(track_bbs_ids)
        if cam1.any():
            ry = 0
            rz = 0
            tx = 0
            ty = 0
            tz = 0.05
            rx = 1.65
            img, cam_arr = render_lidar_on_image(arr, cam1, rx, ry, rz, tx, ty, tz, 9000, 9000)
            # print(img)
            cv2.imshow('Camera', img)
            cv2.waitKey(1)
            update = 1
        # plt.show()



"""
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
        
        
 

"""