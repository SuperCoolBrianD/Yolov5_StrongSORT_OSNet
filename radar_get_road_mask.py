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
for j, i in enumerate(bag.read_messages()):
    sensor = frame.load_data(i)

    # print(idx)
    # print(sensor)

    if frame.full_data:
        # print(abs(abs(frame.camera.message.header.stamp.to_sec() - frame.radar.message.header.stamp.to_sec()) - 1))
        # print(frame.camera.message.header.stamp.to_sec() - frame.radar.message.header.stamp.to_sec())
        print(frame.radar.message.header.stamp.to_sec())
        # print(frame.radar.message.header.stamp.to_sec()- epoch)
        # epoch = frame.radar.message.header.stamp.to_sec()

        image_np = imgmsg_to_cv2(frame.camera.message)
        npts = frame.radar.message.width
        arr_all = pc2_numpy(frame.radar.message, npts)
        axs.set_xlim(-50, 100)
        axs.set_ylim(-50, 150)
        axs.scatter(arr_all[:, 0], arr_all[:, 1], s=0.5)
        plt.show()
        axs.set_xlim(-50, 100)
        axs.set_ylim(-50, 100)
        hull = np.asarray(klicker.get_positions())
        print(hull)
        input()
        p_arr_all = arr_all.copy()
