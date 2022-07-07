import mayavi.mlab as mlab
import sys

import cv2

sys.path.append('yolor')
sys.path.append('SVSF_Track')
#from yolor.detect_custom import init_yoloR, detect
from SVSF_Track.MTT_Functions import *
from radar_utils import *
from projectutils import draw_radar
import rosbag
from matplotlib.animation import FuncAnimation
from vis_util import *
from mpl_point_clicker import clicker
import matplotlib.path as mpltPath
# Read recording
bag = rosbag.Bag("record/tripod.bag")
# bag = rosbag.Bag("record/traffic1.bag")
topics = bag.get_type_and_topic_info()

# old SORT tracker
# mot_tracker = sort.Sort(min_hits=2, max_age=8, iou_threshold=0.1)

for i in topics[1]:
    print(i)

radar_d = '/radar_data'
s = 0
# model, device, colors, names = init_yoloR(weights='yolor/yolor_p6.pt', cfg='yolor/cfg/yolor_p6.cfg',
#                                           names='yolor/data/coco.names', out='inference/output', imgsz=640)
# adjust image visualization
cv2.namedWindow("Camera")
cv2.moveWindow('Camera', 800, 800)
# init loop


# mtx = np.array([[234.45076996, 0., 334.1804498],
#                 [0.,311.6748573,241.50825294],
#                 [0., 0., 1.]])
# mtx = np.array([[1113.5, 0., 974.2446],
#                 [0.,1113.5,586.6797],
#                 [0., 0., 1.]])
mtx = np.array([[747.9932, 0., 655.5036],
                [0., 746.6126, 390.1168],
                [0., 0., 1.]])




frame = SRS_data_frame()
p_arr_all = np.array([[0, 0, 0, 0, 0]])
figs, axs = plt.subplots(1, figsize=(6, 6))
klicker = clicker(axs, ["event"], markers=["x"], **{"linestyle": "--"})
hull = np.array([[-16.0823566, 16.43046557],
                [38.99491754, 96.56723232],
                [47.08536459, 78.93714363],
                [9.58925423, 25.17265831],
                [11.92303703, 13.80780775],
                [-8.38087335, -6.88204839]])
p_radar = np.empty((0, 5))
for j, i in enumerate(bag.read_messages()):
    sensor = frame.load_data(i)

    # print(idx)
    # print(sensor)

    if frame.full_data:
        # print(abs(abs(frame.camera.message.header.stamp.to_sec() - frame.radar.message.header.stamp.to_sec()) - 1))
        # print(frame.camera.message.header.stamp.to_sec() - frame.radar.message.header.stamp.to_sec())
        figs, axs = plt.subplots(1, figsize=(6, 6))
        image_np = imgmsg_to_cv2(frame.camera.message)
        npts = frame.radar.message.width
        arr_all = pc2_numpy(frame.radar.message, npts)
        axs.scatter(arr_all[:, 0], arr_all[:, 1], s=0.5, c='red')
        arr_concat = np.vstack((arr_all, p_radar))
        path = mpltPath.Path(hull)
        mask = path.contains_points(arr_concat[:, :2])
        arr = arr_concat[mask]
        # axs.scatter(arr_masked[:, 0], arr_masked[:, 1], s=0.5)

        axs.set_xlim(-50, 100)
        axs.set_ylim(-50, 100)
        # arr = filter_zero(arr_all)
        axs.scatter(arr[:, 0], arr[:, 1], s=0.5)
        # draw points on plt figure
        pc = arr[:, :4]
        ped_box = np.empty((0, 5))
        # Perform class specific DBSCAN
        total_box, cls = dbscan_cluster(pc, eps=2.5, min_sample=15, axs=axs)
        cv2.imshow('0', image_np)
        cv2.waitKey(1)
        p_radar = arr_all.copy()
        plt.show()
