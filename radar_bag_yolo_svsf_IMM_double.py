import sys
sys.path.append('yolor')
sys.path.append('SVSF_Track')
from radar_utils import *
from yolor.detect_custom import init_yoloR, detect
from SVSF_Track.MTT_Functions import *
import pickle
import rosbag
from matplotlib.animation import FuncAnimation
from learn_util import *
from auto_label_util import *
import time
import open3d as o3d
# Read recording
bag = rosbag.Bag("record/tripod.bag")
# bag = rosbag.Bag("record/traffic3.bag")
# bag = rosbag.Bag("record/traffic1.bag")
topics = bag.get_type_and_topic_info()

# old SORT tracker
# mot_tracker = sort.Sort(min_hits=2, max_age=8, iou_threshold=0.1)

for i in topics[1]:
    print(i)

# init plt figure
fig, axs = plt.subplots(1, figsize=(6, 6))
fig.canvas.set_window_title('Radar Detection and Tracking IMM_small')
# create generator object for recording
bg = bag.read_messages()
# plotting resolution
s = 0
half = False
# model, device, colors, names = init_yoloR(weights='yolor/yolor_p6.pt', cfg='yolor/cfg/yolor_p6.cfg',
#                                           names='yolor/data/coco.names', out='inference/output', imgsz=1280, half=False)
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
tracked_object_radar = dict()
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
models = ["CV", 'CA']
# models = ["CV"]
filters = ['IPDAKF', 'IPDAKF']
# filters = ['IPDAKF']
G_List = [G_CV, G_CA] #input gain list
Q_List = [Q_CV, Q_CA] #process noise co-variance list
# G_List = [G_CV] #input gain list
# Q_List = [Q_CV, Q_CT] #process noise co-variance list
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


mtx = np.array([[748, 0., 655.5],
                [0.,746.6,390.11],
                [0., 0., 1.]])
cam_msg = 0
# mtx = np.array([[234.45076996, 0., 334.1804498],
#                 [0.,311.6748573,241.50825294],
#                 [0., 0., 1.]])
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
classes = ['car', 'bus', 'person', 'truck', 'no_match']
rx = 1.56
ry = 0
rz = -.02
tx = 0
ty = 0
tz = 0.1

r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)
frame = SRS_data_frame()
idx=0


hull = np.array([[-18.06451613,  17.53246753],
       [ 53.87096774, 114.93506494],
       [ 63.87096774, 102.81385281],
       [ 14.51612903,  27.92207792],
       [ 10.96774194,   9.74025974],
       [ -1.61290323,  -1.94805195],
        [-18.06451613,  17.53246753]])
p_radar = np.empty((0, 5))

def animate(g):
    global image_np
    global framei
    global imgs
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
    global hull
    global p_radar
    i = next(bg)

    # read ros Topic camera or radar
    sensor = frame.load_data(i)
    # print(idx)
    # print(sensor)


    if frame.full_data:
        # print(abs(abs(frame.camera.message.header.stamp.to_sec() - frame.radar.message.header.stamp.to_sec()) - 1))
        # print(frame.camera.message.header.stamp.to_sec()- epoch)
        plt.cla()
        plt.plot(hull[:, 0], hull[:, 1], 'k-')
        image_np = imgmsg_to_cv2(frame.camera.message)
        npts = frame.radar.message.width
        arr_all = pc2_numpy(frame.radar.message, npts)
        axs.scatter(arr_all[:, 0], arr_all[:, 1], s=0.5, c='red')
        arr_concat = np.vstack((arr_all, p_radar))
        path = mpltPath.Path(hull)
        mask = path.contains_points(arr_concat[:, :2])
        arr = arr_concat[mask]
        idx+=1
        # draw points on plt figure
        axs.set_xlim(-50, 100)
        axs.set_ylim(-50, 100)
        # arr = filter_zero(arr_all)
        axs.scatter(arr[:, 0], arr[:, 1], s=0.5)
        # draw points on plt figure

        pc = arr[:, :4]
        # Perform class specific DBSCAN
        total_box, cls = dbscan_cluster(pc, eps=2.5, min_sample=15, axs=axs)
        total_box_1, cls_1 = dbscan_cluster(pc, eps=2, min_sample=6, axs=axs)
        total_box = np.vstack((total_box, total_box_1))
        box_index = non_max_suppression_fast(total_box[:, :4], .2)
        cls.extend(cls_1)
        cls = [cls[ii] for ii in box_index]
        total_box = total_box[box_index, :]
        for ii, jj in enumerate(cls):
            plot_box(total_box[ii, :], jj, axs)
        cv2.imshow('0', image_np)
        cv2.waitKey(1)
        # Do pedestrian if required
        # ped_box, ped_cls = dbscan_cluster(pc, eps=2, min_sample=10)

        # measSet = np.empty((0, 11))
        # # KF tracking
        #
        # if cls:
        #     # convert clusters into tracking input format
        #     for ii in cls:
        #         bbox = get_bbox_cls(ii)
        #         x = np.hstack((np.mean(ii, axis=0), bbox))
        #         measSet = np.vstack((measSet, x))
        #     measSet = measSet.T[:2, :]
        #     # perform tracking
        #     trackList, unassignedMeas = gating(trackList, lastTrackIdx, PG, MP_IMM, maxVals, sensorPos, measSet)
        #     # perform gating
        #     trackList = updateStateTracks(trackList,lastTrackIdx, filterType, measSet, maxVals,
        #                                   lambdaVal,MP_IPDA, PG, PD, sensorPos, SVSFParams, k)
        #     # update the state of each track
        #     trackList = updateTracksStatus(trackList,lastTrackIdx, delTenThr, delConfThr, confTenThr,k)
        #     # update the status of each track usiing the track manager
        #     # initiate tracks for measurements that were not gated or in other words unassigned measurements
        #     trackList, lastTrackIdx = initiateTracksMM(trackList,lastTrackIdx, unassignedMeas, maxVals, G_List, H,
        #                                                Q_List, R, models,filters, Ts, pInit, k, sensor, N)
        #     """
        #     each frame -> measSet [x, y] (change this in to x, y, length, width)
        #
        #     tracklist
        #
        #     a(x, y, l, w) b c d
        #
        #     tracklist (trackobj1(x, y trackobj2 ....)
        #
        #     unassignment measurement = [a, b]
        #     gated measurement = [c, d]
        #     """
        #
        #     #input("")
        #     k += 1
        #     # iterate through all tracks
        #     # ToDo change tracklet format so it doesn't need to iterate through all previous tracks
        #     for jj, ii in enumerate(trackList[:lastTrackIdx]):
        #         # print(ii.BLs)
        #         # get centroid
        #
        #         centroid = ii.xPost[:2]
        #         speed = np.sqrt(ii.xPost[2]**2 + ii.xPost[3]**2)
        #
        #         latency = ii.latency
        #         pCurrent = ii.pCurrent
        #
        #
        #         status = ii.status
        #         # if track has terminated remove from tracked objects
        #         if ii.endSample:
        #             tracked_object[jj] = None
        #             alive_track[jj] = False
        #             continue
        #         # add if first appearance
        #         if jj not in tracked_object.keys():
        #             tracked_object[jj] = radar_object((centroid[0], centroid[1]))
        #             alive_track[jj] = True
        #         # update if in tracked objects
        #         elif jj in tracked_object.keys() and tracked_object[jj]:
        #             tracked_object[jj].upd((centroid[0], centroid[1]))
        #             axs.text(centroid[0], centroid[1] - 2, 'ID: ' + str(jj), fontsize=11,
        #                      color='r')
        #             axs.text(centroid[0], centroid[1] - 5, 'Speed: ' + f'{speed*3.6:.2f} km/h', fontsize=11,
        #                      color='r')
        #             axs.text(centroid[0], centroid[1] - 12, 'Prob: ' + f'{pCurrent:.2f}', fontsize=11,
        #                      color='r')
        #
        #             # if latency >=0:
        #             #    axs.text(centroid[0], centroid[1] - 8, 'Latency: ' + f'{latency*Ts:.2f} s', fontsize=11,
        #             #             color='r')
        #
        #     # plot and update all alive tracks
        #
        #     for tk, j in enumerate(alive_track[:lastTrackIdx]):
        #         if j:
        #             tracked_object[tk].life -= 1
        #             x = [obtk[0] for obtk in tracked_object[tk].tracklet]
        #             y = [obtk[1] for obtk in tracked_object[tk].tracklet]
        #             axs.plot(x, y, mew=0)
        #             if tracked_object[tk].life == 0:
        #                 alive_track.remove(tk)
        # # yolo detection
        # # cam1, detection = detect(source=cam1, model=model, device=device, colors=colors, names=names,
        # #                              view_img=False)
        # # Radar projection onto camera parameters
        # ttt = time.time()
        # # image_np_detection, detection = detect(source=image_np, model=model, device=device, colors=colors, names=names,
        # #                          view_img=False, half=half)
        # if cls:
        #     for cc in cls:
        #         bbox = get_bbox_cls(cc)
        #         # print(bbox)
        #         bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)
        #         bbox = project_to_image(bbox, r2c)
        #         pts = project_to_image(cc.T, r2c)
        #         # print(pts)
        #         box2d = get_bbox_2d(pts.T)
        #         # draw_projected_box3d(image_np_detection, bbox)
        #         # cv2.rectangle(image_np_detection, box2d[0], box2d[1], (255, 255, 0))
        #         # cv2.rectangle(image_np_detection, box2d[0], box2d[1], (255, 255, 0))
        #         # box2d = [box2d[0][0], box2d[0][1], box2d[1][0], box2d[1][1]]
        #         # box2d = convert_topy_bottomx(box2d)
        #         # matched = find_gt(box2d, detection)
        #         # label = get_bbox_cls_label(cc, matched[0][1])
        #         # # bbframe.append(bbox)
        #         # file.write(' '.join([str(num) for num in label]))
        #         # file.write('\n')
        #
        #         # draw_projected_box3d(cam1, bbox)
        # # img, cam_arr = render_radar_on_image(arr_all, cam1, r2c, 9000, 9000)
        # # cv2.imshow('Camera', image_np)
        # # cv2.waitKey(1)
        # update = 1
        # # file.close()
        p_arr_all = arr_all.copy()



ani = FuncAnimation(fig, animate, interval=10, frames=2000)
plt.show()


"""
store all detected object in the previous frame
iterate through all object in the previous frame
if the object is tracked give it orientation
rotate the object 
find l w h 

"""