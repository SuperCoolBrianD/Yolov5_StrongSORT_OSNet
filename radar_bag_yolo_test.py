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

radar_d = '/radar_data'
# init plt figure
fig, axs = plt.subplots(1, figsize=(6, 6))
fig.canvas.set_window_title('Radar Detection and Tracking IMM_small')
# create generator object for recording
bg = bag.read_messages()
s = 0
model, device, colors, names = init_yoloR(weights='yolor/yolor_p6.pt', cfg='yolor/cfg/yolor_p6.cfg',
                                           names='yolor/data/coco.names', out='inference/output', imgsz=1280)
# adjust image visualization
cv2.imshow('Camera', np.zeros((480, 640)))
cv2.moveWindow('Camera', 800, 800)

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


mtx = np.array([[748, 0., 655.5],
                [0.,746.6,390.11],
                [0., 0., 1.]])
cam_msg = 0
image_np = np.empty(shape=(5, 5))
print(image_np)
for j, i in enumerate(bg):
    print(i.topic)
    # read ros Topic camera or radar

    if i.topic == '/Camera':
        print('hello')
        cam_msg = i.message.header.stamp.to_sec()
        image_np = imgmsg_to_cv2(i.message)
    elif i.topic == '/Radar':
        npts = i.message.width
        arr_all = pc2_numpy(i.message, npts)
        tm = i.message.header.stamp.to_sec()
        arr = filter_zero(arr_all)
        pc = arr[:, :4]
        ped_box = np.empty((0, 5))
        # Perform class specific DBSCAN
        total_box, cls = dbscan_cluster(pc, eps=3, min_sample=15, axs=axs)
        if total_box.any() and ped_box.any:
            total_box = np.vstack((total_box, ped_box))
        if image_np.shape != (5, 5):
            # yolo detection
            cam1, detection = detect(source=image_np, model=model, device=device, colors=colors, names=names,
                                         view_img=False)
            cv2.imshow('0', cam1)
            cv2.waitKey(1)

