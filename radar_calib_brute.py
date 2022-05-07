import mayavi.mlab as mlab
import rospy
import rosbag
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Int32, String
import numpy as np
import cv2
from projectutils import draw_radar
from retina_view.msg import MsgRadarPoint
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from radar_utils import *
# Press the green button in the gutter to run the script.
bag = rosbag.Bag("record/traffic3.bag")
topics = bag.get_type_and_topic_info()
# print(topics)

for i in topics[1]:
    print(i)
# print(type(bag))
# fig = mlab.figure(size=(500, 500), bgcolor=(0, 0, 0))
radar_d = '/radar_data'
view=(180.0, 70.0, 250.0, ([12.0909996 , -1.04700089, -2.03249991]))
image_np = np.empty(0)

mtx = np.array([[381, 0., 324],
                [0.,400, 265],
                [0., 0., 1.]])

# mtx = np.array([[234.45076996, 0., 334.1804498],
#                 [0.,311.6748573,241.50825294],
#                 [0., 0., 1.]])


dist = np.array([[-0.06624252, -0.00059409, -0.00183169,  0.0030411,   0.00063524]])

h,  w = 480, 640
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))



for j, i in enumerate(bag.read_messages()):
    # print(i.topic)
    if j < 150:
        continue
    if i.topic == '/usb_cam/image_raw/compressed':
        np_arr = np.frombuffer(i.message.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        print(image_np.shape)
    elif i.topic == '/radar_data':
        # print(type(i.message.points[0]))
        # print(i.message.points[0])
        arr = convert_to_numpy(i.message.points)
        arr = filter_zero(arr)
        # draw_radar(arr, fig=fig)

        if image_np.any():
            print('start')

            # rx = float(input('Input rx'))
            # ry = float(input('Input ry'))
            # rz = float(input('Input rz'))
            # tx = float(input('Input tx'))
            # ty = float(input('Input ty'))
            # tz = float(input('Input tz'))
            ry = 0
            rz = 0
            tx = -0.05
            ty = 0
            tz = 0.05
            rx = 1.57
            r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)

            img, cam_arr = render_radar_on_image(arr, image_np, r2c, 9000, 9000)
            cv2.imshow('0', img)
            cv2.waitKey(50)
            # mlab.clf(fig)
        pc = arr[:, :4]

        # pc = StandardScaler().fit_transform(pc)
        # clustering = DBSCAN(eps=2, min_samples=40)
        # clustering.fit(pc)
        # label = clustering.labels_
        # cls = set_cluster(pc, label)
        # mlab.show(stop=True)

    # input()