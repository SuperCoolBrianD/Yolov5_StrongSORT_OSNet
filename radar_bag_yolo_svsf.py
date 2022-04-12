import sys
sys.path.append('yolor')
sys.path.append('SVSF_Track')
from SVSF_Track.MTT_Functions import *
from radar_utils import *
from yolor.detect_custom import init_yoloR, detect
import sort
import rosbag
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import torch
import time


bag = rosbag.Bag("record/traffic1.bag")
# bag = rosbag.Bag("record/traffic1.bag")
topics = bag.get_type_and_topic_info()
# print(topics)

mot_tracker = sort.Sort(min_hits=2, max_age=8, iou_threshold=0.1)

for i in topics[1]:
    print(i)
# print(type(bag))
# fig = mlab.figure(size=(1000, 1000), bgcolor=(0, 0, 0))
radar_d = '/radar_data'
fig, axs = plt.subplots(1, figsize=(6, 6))
bg = bag.read_messages()
s = 0
# model, device, colors, names = init_yoloR(weights='yolor/yolor_p6.pt', cfg='yolor/cfg/yolor_p6.cfg',
#                                           names='yolor/data/coco.names', out='inference/output', imgsz=640)
cv2.imshow('Camera', np.zeros((480, 640)))
cv2.moveWindow('Camera', 800, 800)
image_np = np.empty((5, 5))
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))
framei = 0
imgs = []
radar_time_0 = 0
camera_time_0 = 0
update = 1
cam1 = np.empty((5,5))
tracked_object = dict()
alive_track = [0] * 8000
life = 10
# tack parameters
trackList = [0] * 8000  # allocate track list, a list of objects
lastTrackIdx = -1  # index of a track in the list,
# init KF tracker
sigma_v = 1E-1 #process noise standard deviation
sigma_v_filt = sigma_v #process noise standard deviation for filter
sigma_w = .5 #measurement noise standard deviation in position
R = np.diag(np.array([sigma_w ** 2, sigma_w ** 2]))  # measurement co-variance
H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])  # measurement matrix
pInit = .2 #initial probability of track existence
PG = .999 #gate probability
Ts = .12 #sampling time
unassignedMeas0 = np.array([[]])
maxVel = 16 #for initializing velocity variance for 1-point initialization
omegaMax = math.radians(4) #for initializing turn-rate variance for 1-point initialization
modelType = 'CT'
sensor = 'Lidar'
filterType = "IPDAKF"
sensorPos = np.array([0, 0])
P_ii = .95  #for Markov matrix of IPDA
PD = .6 #probability of target detection in a time step
MP = np.array([[P_ii,1-P_ii],[1-P_ii,P_ii]]) #Markov matrix for IPDA
lambdaVal = 0.05 # parameter for clutter density
delTenThr = .05 #threshold for deleting a tenative track
confTenThr = .9 # threshold for confirming a tenative track
delConfThr = 0.05 # threshold for deleting a confirmed track

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


N = 1000
trackList,lastTrackIdx = initiateTracks(trackList,lastTrackIdx, unassignedMeas0, sigma_w, maxVel, omegaMax, G, H, Q, R,
                                        modelType, Ts, pInit, 0, sensor, N)
k = 0


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
    # print(framei)
    # to stop after certain amount of frames
    i = next(bg)
    # print(i.topic)
    if i.topic == '/usb_cam/image_raw/compressed':
        # if camera_time_0 == 0:
        #     camera_time_0 = i.message.header.stamp

        np_arr = np.frombuffer(i.message.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if update:
            cam1 = image_np
            update = 0
        for tk, j in enumerate(alive_track[:lastTrackIdx]):
            if j:
                # print(tracked_object[tk].tracklet)
                x = [obtk[0][0] for obtk in tracked_object[tk].tracklet]
                y = [obtk[0][1] for obtk in tracked_object[tk].tracklet]
                axs.plot(x, y, mew=0)

    # print(i.message.header)
        # imgs.append(image_np)
    elif i.topic == '/radar_data' or 'radar_data':
        # if radar_time_0 == 0:
        plt.cla()
        tm = i.message.time_stamp[0]/10**6
        axs.set_xlim(-50, 80)
        axs.set_ylim(0, 100)
        # print(type(i.message.points[0]))
        # print(i.message.points[0])
        arr = convert_to_numpy(i.message.points)
        arr = filter_zero(arr)
        axs.scatter(arr[:, 0], arr[:, 1], s=0.5)
        pc = arr[:, :4]
        ped_box = np.empty((0, 5))
        # pc = StandardScaler().fit_transform(pc)
        total_box, cls = dbscan_cluster(pc, eps=2, min_sample=20, axs=axs)
        if total_box.any() and ped_box.any:
            total_box = np.vstack((total_box, ped_box))
        # ped_box, ped_cls = dbscan_cluster(pc, eps=2, min_sample=10)
        measSet = np.empty((0, 4))
        if cls:
            for ii in cls:
                # print(ii)
                measSet = np.vstack((measSet, np.mean(ii, axis=0)))
            measSet = measSet.T[:2, :]

            trackList, unassignedMeas = gating(trackList, lastTrackIdx, PG, sensorPos, measSet)
            # perform gating
            trackList = updateStateTracks(trackList,lastTrackIdx, filterType, measSet, lambdaVal,MP, PG, PD, sensorPos, k)
            # update the state of each track
            trackList = updateTracksStatus(trackList,lastTrackIdx, delTenThr, delConfThr, confTenThr,k)
            # update the status of each track usiing the track manager
            # initiate tracks for measurements that were not gated or in other words unassigned measurements
            trackList, lastTrackIdx = initiateTracks(trackList,lastTrackIdx,unassignedMeas, sigma_w, maxVel, omegaMax, G, H, Q, R, modelType, Ts, pInit, k, sensor, N)
            for jjj in trackList:
                if jjj==0:
                    break
                # print(jjj.pCurrent)
            k += 1
            for jj, ii in enumerate(trackList[:lastTrackIdx]):
                # print(ii.xPost)
                centroid = ii.xPost[:2]

                if ii.endSample:
                    tracked_object[jj] = None
                    alive_track[jj] = False
                    continue

                if jj not in tracked_object.keys():
                    tracked_object[jj] = radar_object(((centroid[0], centroid[1]), tm))
                    alive_track[jj] = True
                elif jj in tracked_object.keys() and tracked_object[jj]:
                    tracked_object[jj].upd(((centroid[0], centroid[1]), tm))
                    axs.text(centroid[0], centroid[1] - 2, 'ID: ' + str(jj), fontsize=11,
                             color='r')
            print([iii.pCurrent for iii in trackList[:lastTrackIdx]])
            print([iii.endSample for iii in trackList[:lastTrackIdx]])
            print([iii.status for iii in trackList[:lastTrackIdx]])
            for tk, j in enumerate(alive_track[:lastTrackIdx]):
                if j:
                    tracked_object[tk].life -= 1
                    # print(tracked_object[tk].tracklet)
                    x = [obtk[0][0] for obtk in tracked_object[tk].tracklet]
                    y = [obtk[0][1] for obtk in tracked_object[tk].tracklet]
                    axs.plot(x, y, mew=0)
                    if tracked_object[tk].life == 0:
                        alive_track.remove(tk)
            # print(alive_track)
                # if tracked_object[jj].speed == 'Measuring':
                #     axs.text(centroid[0], centroid[1] - 5,
                #              f'Speed: Measuring', fontsize=11,
                #              color='r')
                # else:
                #     axs.text(centroid[0], centroid[1] - 5, f'Speed: {float(tracked_object[jj].speed):.2f} km/h' , fontsize=11,
                #          color='r')

            # track_bbs_ids = mot_tracker.update(total_box)

            # print(alive_track)
            # print(track_bbs_ids)
        if cam1.any():
            # cam1, detection = detect(source=cam1, model=model, device=device, colors=colors, names=names,
            #                              view_img=False)
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


ani = FuncAnimation(fig, animate, interval=10, frames=2000)
plt.show()


"""
Key issue

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