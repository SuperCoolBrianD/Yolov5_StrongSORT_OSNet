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


def filter_cluster(pc, label):
    msk = label != -1
    pc = pc[msk, :]
    if pc.shape[0] == 0:
        return np.zeros([1,5])

    # print(msk)
    return pc


def set_cluster(pc, label):
    l = np.unique(label)
    pc_list = []
    for i in l:
        if i != -1:
            msk = label == i
            pts = pc[msk, :]
            pc_list.append(pts)
    if not pc_list:
        return [np.zeros([1,5])]
    return pc_list


def convert_to_numpy(pc):
    l = len(pc)
    arr = np.zeros((l, 5))
    for i, point in enumerate(pc):
        arr[i, 0] = point.x
        arr[i, 1] = point.y
        arr[i, 2] = point.z
        arr[i, 3] = point.doppler
    return arr


def filter_zero(pc):
    mask = np.abs(pc[:, 3]) > 0.05
    s = np.sum(mask)
    # print(pc.shape)
    # print(mask)
    # print(mask.shape)
    pc = pc[mask, :]
    if pc.shape[0] == 0:
        return np.zeros([1,5])
    return pc


bag = rosbag.Bag("record/car_1.bag")
topics = bag.get_type_and_topic_info()
# print(topics)

for i in topics[1]:
    print(i)
# print(type(bag))
fig = mlab.figure(size=(1000, 1000), bgcolor=(0, 0, 0))
radar_d = '/radar_data'
view=(180.0, 70.0, 250.0, ([12.0909996 , -1.04700089, -2.03249991]))
image_np = np.empty(0)
for j, i in enumerate(bag.read_messages()):

    if j < 350:
        continue
    mlab.clf(fig)
    if i.topic == '/usb_cam/image_raw/compressed':
        np_arr = np.frombuffer(i.message.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        cv2.imshow('0', image_np)
        cv2.waitKey(1)
    elif i.topic == '/radar_data':
        # print(type(i.message.points[0]))
        # print(i.message.points[0])
        arr = convert_to_numpy(i.message.points)
        draw_radar(arr, fig=fig, view=view, pts_scale=0.3)
        arr = filter_zero(arr)

        if image_np.any():
            img, arr = render_lidar_on_image(arr, image_np, np.pi/2, 0, 0, 0, 0, 0, 9000, 9000)
            print(img.shape)


        #
        pc = arr[:, :4]

        # pc = StandardScaler().fit_transform(pc)
        clustering = DBSCAN(eps=2, min_samples=40)
        clustering.fit(pc)
        label = clustering.labels_
        cls = set_cluster(pc, label)


        # print(cls)
        # pc = filter_cluster(pc, label)
        # draw_radar(arr, fig=fig)
        # for c in cls:
        #     draw_radar(c, fig=fig, pts_color=(np.random.random(), np.random.random(), np.random.random()),
        #                          view=view)
        view = mlab.view()

        mlab.show(stop=True)

    # input()
