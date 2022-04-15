import numpy as np
import sys
sys.path.append('yolor')
from yolor.detect_custom import init_yoloR, detect
import sort
import rosbag
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from radar_utils import *

bag = rosbag.Bag("record/car_1.bag")
topics = bag.get_type_and_topic_info()
# print(topics)

mot_tracker = sort.Sort(min_hits=3, max_age=8, iou_threshold=0.1)

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
alive_track = []
life = 10





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
    # print(framei)
    # to stop after certain amount of frames
    framei+=1
    if framei == 2000:
        out.release()
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
        # ped_box, ped_cls = dbscan_cluster(pc, eps=2, min_sample=10)
        if total_box.any() and ped_box.any:
            total_box = np.vstack((total_box, ped_box))
        track_bbs_ids = mot_tracker.update(total_box)
        for t in range(track_bbs_ids.shape[0]):
            centroid = track_bbs_ids[t, :4]
            if track_bbs_ids[t, -1] not in tracked_object.keys():
                alive_track.append(track_bbs_ids[t, -1])
                # if new object add object to all tracklet
                tracked_object[int(track_bbs_ids[t, -1])] = radar_object(((centroid[0], centroid[1]), tm))
            else:
                tracked_object[int(track_bbs_ids[t, -1])].upd(((centroid[0], centroid[1]), tm))
                tracked_object[int(track_bbs_ids[t, -1])].update_speed()

            axs.text(track_bbs_ids[t, 0], track_bbs_ids[t, 1] - 2, 'ID: '+str(int(track_bbs_ids[t, -1])), fontsize=11,
                     color='r')
            if tracked_object[int(track_bbs_ids[t, -1])].speed == 'Measuring':
                axs.text(track_bbs_ids[t, 0], track_bbs_ids[t, 1] - 5,
                         f'Speed: Measuring', fontsize=11,
                         color='r')
            else:
                axs.text(track_bbs_ids[t, 0], track_bbs_ids[t, 1] - 5, f'Speed: {float(tracked_object[int(track_bbs_ids[t, -1])].speed):.2f} km/h' , fontsize=11,
                     color='r')

        for tk in alive_track:
            tracked_object[tk].life -= 1
            x = [obtk[0][0] for obtk in tracked_object[tk].tracklet]
            y = [obtk[0][1] for obtk in tracked_object[tk].tracklet]
            axs.plot(x, y, mew=0)
            if tracked_object[tk].life == 0:
                alive_track.remove(tk)
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

ani = FuncAnimation(fig, animate, interval=1, frames=2000)
plt.show()


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