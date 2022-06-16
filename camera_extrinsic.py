import sys
sys.path.append('yolor')
sys.path.append('SVSF_Track')
from radar_utils import *
from yolor.detect_custom import init_yoloR, detect
from SVSF_Track.MTT_Functions import *
import rosbag
from matplotlib.animation import FuncAnimation

# Read recording
bag = rosbag.Bag("record/car.bag")
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
fig.canvas.set_window_title('Radar Detection and Tracking IMM_small')
# create generator object for recording
bg = bag.read_messages()
s = 0
# model, device, colors, names = init_yoloR(weights='yolor/yolor_p6.pt', cfg='yolor/cfg/yolor_p6.cfg',
#                                            names='yolor/data/coco.names', out='inference/output', imgsz=1280)
# adjust image visualization
cv2.imshow('Camera', np.zeros((480, 640)))
cv2.moveWindow('Camera', 800, 800)
image_np = np.empty((5, 5))



mtx = np.array([[770, 0., 639.5],
                [0.,770,381.6629],
                [0., 0., 1.]])
cam_msg = 0
frame = SRS_data_frame()
ry = 0
rz = .01
tx = 0.7
ty = 0.01
tz = 0
rx = 1.58
r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)



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
    global cam_msg
    global frame
    i = next(bg)
    # read ros Topic camera or radar
    sensor = frame.load_data(i)
    if frame.full_data:
        plt.cla()
        axs.set_xlim(-50, 80)
        axs.set_ylim(-50, 150)
        print(abs(abs(frame.camera.message.header.stamp.to_sec() - frame.radar.message.header.stamp.to_sec()) - 1))
        # print(frame.camera.message.header.stamp.to_sec()- epoch)
        image_np = imgmsg_to_cv2(frame.camera.message)
        cam1 = image_np
        npts = frame.radar.message.width
        arr_all = pc2_numpy(frame.radar.message, npts)[:2, :]
        print(arr_all)
        # draw points on plt figure
        arr = filter_zero(arr_all)

        axs.scatter(arr_all[:, 0], arr_all[:, 1], s=4)

        # draw points on plt figure

        pc = arr[:, :4]
        ped_box = np.empty((0, 5))
        # Perform class specific DBSCAN
        total_box, cls = dbscan_cluster(pc, eps=2, min_sample=20, axs=axs)
        # Do pedestrian if required
        # ped_box, ped_cls = dbscan_cluster(pc, eps=2, min_sample=10)

        if total_box.any() and ped_box.any:
            total_box = np.vstack((total_box, ped_box))
        # cam1, detection = detect(source=cam1, model=model, device=device, colors=colors, names=names,
        #                              view_img=False)
        # Radar projection onto camera parameters

        if cls:
            for cc in cls:
                bbox = get_bbox_cls(cc)
                # print(bbox)
                bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)
                bbox = project_to_image(bbox, r2c)
                pts = project_to_image(cc.T, r2c)
                # print(pts)
                box2d = get_bbox_2d(pts.T)
                cv2.rectangle(cam1, box2d[0], box2d[1], (255,255,0))
                # draw_projected_box3d(cam1, bbox)
        img, cam_arr = render_radar_on_image(arr_all, cam1, r2c, 9000, 9000)
        cv2.imshow('Camera', img)
        cv2.waitKey(1)
        input()
        # update = 1
        # # print(f'Max dt = {max(dt_array)}')
        # # print(f'Min dt = {min(dt_array)}')
        # # print(f'Average dt = {sum(dt_array)/len(dt_array)}')
        # # print(f'Max dt_no_match = {max(dt_no_match_array)}')
        # # print(f'Min dt_no_match = {min(dt_no_match_array)}')
        # # print(f'Average dt_no_match = {sum(dt_no_match_array)/len(dt_no_match_array)}')


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