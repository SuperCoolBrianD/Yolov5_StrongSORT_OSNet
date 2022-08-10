import sys
sys.path.append('yolor')
sys.path.append('SVSF_Track')
from radar_utils import *
import rosbag
from mpl_point_clicker import clicker

# Read recording
bag = rosbag.Bag("record/rooftop.bag")
# bag = rosbag.Bag("record/traffic1.bag")
topics = bag.get_type_and_topic_info()

cv2.namedWindow("Camera")
cv2.moveWindow('Camera', 800, 800)


# mtx = np.array([[234.45076996, 0., 334.1804498],
#                 [0.,311.6748573,241.50825294],
#                 [0., 0., 1.]])
# mtx = np.array([[1113.5, 0., 974.2446],
#                 [0.,1113.5,586.6797],
#                 [0., 0., 1.]])
mtx = np.array([[747.9932, 0., 655.5036],
                [0., 746.6126, 390.1168],
                [0., 0., 1.]])


hull = np.array([[ -3.87096774,  68.18181818],
       [-17.74193548,  54.32900433],
       [  1.93548387,  16.23376623],
       [-18.70967742,  -1.94805195],
       [ -4.19354839, -19.26406926],
       [ 50.32258065,  24.45887446],
       [ 37.41935484,  37.44588745],
       [ 18.70967742,  29.22077922]])

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



for j, i in enumerate(bag.read_messages()):
    sensor = frame.load_data(i)
    print(frame.full_data)
    if frame.full_data:
        figs, axs = plt.subplots(1, figsize=(6, 6))
        # axs.plot(hull[:, 0], hull[:, 1], 'k-')
        klicker = clicker(axs, ["event"], markers=["x"], **{"linestyle": "--"})
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
        axs.set_xlim(-100, 100)
        axs.set_ylim(-100, 100)
        axs.scatter(arr_all[:, 2], arr_all[:, 0], s=0.5, c='red')
        axs.scatter(arr_non_zero[:, 2], arr_non_zero[:, 0], s=1)
        plt.show()
        axs.set_xlim(-50, 100)
        axs.set_ylim(-50, 100)
        hull = np.asarray(klicker.get_positions())
        print(hull)

        cv2.imshow('camera', image_np)
        cv2.waitKey(1)
        # input()
        p_arr_all = arr_all.copy()
