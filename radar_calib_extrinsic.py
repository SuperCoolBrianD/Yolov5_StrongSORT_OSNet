# import matplotlib
# import mayavi.mlab as mlab
import sys

import cv2

sys.path.append('yolor')
sys.path.append('SVSF_Track')
from radar_utils import *
import rosbag
# from vis_util import *
import pickle

# Read recording
bag = rosbag.Bag("record/rooftop.bag")
# bag = rosbag.Bag("record/traffic1.bag")
topics = bag.get_type_and_topic_info()

# fig = mlab.figure(size=(1000, 500), bgcolor=(0, 0, 0))
calib_pts = []
cam1 = np.empty((0,0))
v = (-108.20802358222203, 7.280529894768495, 470.76425650815855, ([12.091, -1.047, -2.0325]))

rx = 1.56
ry = 0
rz = 0.050000000000000044
tx = 0.2999999999999998
ty = 0
tz = 0


# rx = 1.6
# ry = 0
# rz = .04
# tx = 0
# ty = 0
# tz = 0


r2c_e = extrinsic_matrix(rx, ry, rz, tx, ty, tz)
frame = SRS_data_frame()
mes = []
for j, i in enumerate(bag.read_messages()):
    if j == 1000:
        break
    sensor = frame.load_data(i)
    if frame.full_data:
        image_np = imgmsg_to_cv2(frame.camera.message)
        npts = frame.radar.message.width
        arr_all_rad = pc2_numpy(frame.radar.message, npts)
        arr_all = transform_radar(arr_all_rad.T, r2c_e).T
        moving = filter_zero(arr_all)
        pc = moving[:, :4]
        total_box_1, cls_1 = dbscan_cluster(pc, eps=2, min_sample=2)
        if cls_1:
            num = np.random.uniform()
            if num <= 0.5:
                for ii in cls_1:
                    xyz = np.mean(ii, axis=0)
                    x = xyz[0]
                    z = xyz[2]
                    miny = ii[1, :][np.argmin(ii[1, :])]
                    rng = math.sqrt(z**2 + miny**2)
                    data = [x, miny, z, rng]
                    mes.append(data)
        # draw_radar(arr_all, fig=fig, pts_scale=0.5, pts_color=(1, 0, 1), view=v)
        # draw_radar(arr_all_rad, fig=fig, pts_scale=0.5, pts_color=(1, 1, 1), view=v)
        # draw_radar(moving, fig=fig, pts_scale=0.8, pts_color=(1, 0, 1), view=v)
        cv2.imshow('camera', image_np)
        cv2.waitKey(1)
        # mlab.show(stop=True)
        # print(cls_1)
        # print(moving)
        update = 1
pickle.dump(mes, open('extrinsic_meas.pkl', 'wb'))