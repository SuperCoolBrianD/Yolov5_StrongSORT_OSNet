import os

from radar_utils import *
import sys
sys.path.append('yolor')
sys.path.append('SVSF_Track')
from yolor.detect_custom import init_yoloR, detect
import rosbag
from auto_label_util import *
import pickle


# Read recording



half = False
radar_d = '/radar_data'
s = 0
model, device, colors, names = init_yoloR(weights='yolor/yolor_p6.pt', cfg='yolor/cfg/yolor_p6.cfg',
                                          names='yolor/data/coco.names', out='inference/output', imgsz=1280, half=False)

# adjust image visualization
cv2.namedWindow("Camera")
cv2.moveWindow('Camera', 800, 800)
# init loop


mtx = np.array([[770, 0., 639.5],
                [0.,770,381.6629],
                [0., 0., 1.]])

calib_pts = []
cam1 = np.empty((0,0))
v = (-108.20802358222203, 7.280529894768495, 470.76425650815855, ([12.091, -1.047, -2.0325]))




bag = rosbag.Bag("record/car.bag")

# print(data)
all_bb = []
idx = 0
missidx = 0

rx = 1.56
ry = 0
rz = -.02
tx = 0
ty = 0
tz = 0.1

r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)
frame = SRS_data_frame()
epoch = None
for j, i in enumerate(bag.read_messages()):
    # read ros Topic camera or radar
    sensor = frame.load_data(i)
    # print(idx)
    # print(sensor)
    if sensor == '/Radar':
        npts = frame.radar.message.width
        arr_all = pc2_numpy(frame.radar.message, npts)
        pickle.dump(arr_all, open(f"dataset/{idx:05d}/radar.pkl", 'wb'))
        idx+=1


print(idx)
print(missidx)