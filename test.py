import matplotlib
import mayavi.mlab as mlab
from radar_utils import *
from projectutils import draw_radar
import rosbag
from matplotlib.animation import FuncAnimation
from vis_util import *
import pickle

# Read recording
bag = rosbag.Bag("record/rooftop.bag")
# bag = rosbag.Bag("record/traffic1.bag")
topics = bag.get_type_and_topic_info()

fig = mlab.figure(size=(1000, 500), bgcolor=(0, 0, 0))
calib_pts = []
cam1 = np.empty((0,0))
v = (180.0, 90.0, 192.5583929570825, ([ 25.82646918, -15.33215332,  32.84004135]))

rx = 1.56
ry = 0
rz = 0.050000000000000044
tx = 0.2999999999999998
ty = 0
tz = 0
r2c_e = extrinsic_matrix(rx, ry, rz, tx, ty, tz)
h = 10.9
alpha =1.2238
rx = -20/180*np.pi
ry = 0
rz = 0
tx = 0
ty = -10.9
tz = 0
c2g = extrinsic_matrix(rx, ry, rz, tx, ty, tz)
print(c2g)
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

        # transform from radar to camera
        arr_all = transform_radar(arr_all_rad.T, r2c_e).T
        arr_all_global = transform_radar(arr_all.T, c2g).T
        # from camera to global

        moving = filter_zero(arr_all_global)
        pc = moving[:, :4]
        total_box_1, cls_1 = dbscan_cluster(pc, eps=2, min_sample=2)
        draw_radar(arr_all_global, fig=fig, pts_scale=0.5, pts_color=(1, 0, 1), view=v)
        draw_radar(moving, fig=fig, pts_scale=0.7, pts_color=(1, 1, 1), view=v)
        mlab.plot3d([0, 0], [10.9, 10.9], [0, 100], color=(1,1,1), tube_radius=0.1)
        # draw_radar(moving, fig=fig, pts_scale=0.8, pts_color=(1, 0, 1), view=v)
        cv2.imshow('camera', image_np)
        cv2.waitKey(1)
        # mlab.show(stop=True)
        mlab.clf()
        v = mlab.view()
        print(v)
        # print(cls_1)
        # print(moving)
        update = 1
pickle.dump(mes, open('extrinsic_meas.pkl', 'wb'))