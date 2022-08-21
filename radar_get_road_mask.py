import sys
sys.path.append('yolor')
sys.path.append('SVSF_Track')
from radar_utils import *
import rosbag
from mpl_point_clicker import clicker

# Read recording
bag = rosbag.Bag("record/remote.bag")
# bag = rosbag.Bag("record/traffic1.bag")
topics = bag.get_type_and_topic_info()
mtx = np.array([[747.9932, 0., 655.5036],
                [0., 746.6126, 390.1168],
                [0., 0., 1.]])


frame = SRS_data_frame()
p_arr_all = np.empty((0, 5))


rx = 1.56
ry = 0
rz = 0.050000000000000044
tx = 0.2999999999999998
ty = 0
tz = 0
r2c_e = extrinsic_matrix(rx, ry, rz, tx, ty, tz)
c2g = extrinsic_matrix(-20/180*np.pi, 0, 0, 0, -10.9, 0)
r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)

total_radar_move = np.empty((0, 5))
total_radar_zero = np.empty((0, 5))

for j, i in enumerate(bag.read_messages()):
    if j == 2000:
        break
    sensor = frame.load_data(i)

    # print(idx)
    # print(sensor)

    if frame.full_data:


        # axs.plot(hull[:, 0], hull[:, 1], 'k-')

        # print(abs(abs(frame.camera.message.header.stamp.to_sec() - frame.radar.message.header.stamp.to_sec()) - 1))
        # print(frame.camera.message.header.stamp.to_sec() - frame.radar.message.header.stamp.to_sec())
        print(frame.radar.message.header.stamp.to_sec())
        # print(frame.radar.message.header.stamp.to_sec()- epoch)
        # epoch = frame.radar.message.header.stamp.to_sec()
        image_np = imgmsg_to_cv2(frame.camera.message)
        npts = frame.radar.message.width
        arr_all = pc2_numpy(frame.radar.message, npts)
        arr_concat = np.vstack((arr_all, p_arr_all))
        # coordinate transformation from radar to global
        arr_c = transform_radar(arr_concat.T, r2c_e).T  # from radar to camera (not pixel)
        arr_g = transform_radar(arr_c.T, c2g).T  # from camera to global
        arr_non_zero = filter_zero(arr_g)
        arr_zero = filter_move(arr_g)
        total_radar_move = np.vstack((total_radar_move, arr_non_zero))
        total_radar_zero = np.vstack((total_radar_zero, arr_zero))
        cv2.imshow('camera', image_np)
        cv2.waitKey(1)
        # input()
        p_arr_all = arr_all.copy()

figs, axs = plt.subplots(1, figsize=(6, 6))
klicker = clicker(axs, ["event"], markers=["x"], **{"linestyle": "--"})

axs.set_xlim(-100, 100)
axs.set_ylim(-100, 100)
axs.scatter(total_radar_zero[:, 0], total_radar_zero[:, 2], s=0.5, color='r')
axs.scatter(total_radar_move[:, 0], total_radar_move[:, 2], s=1)

plt.show()
hull = np.asarray(klicker.get_positions())
print(hull)