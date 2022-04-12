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

# Press the green button in the gutter to run the script.


def filter_cluster(pc, label):
    msk = label != -1
    print(label)
    print(msk)
    return pc[msk, :]


bag = rosbag.Bag("srs.bag")
topics = bag.get_type_and_topic_info()
# print(topics)

for i in topics[1]:
    print(i)
# print(type(bag))
fig = mlab.figure(size=(1000, 1000), bgcolor=(0, 0, 0))
for i in bag.read_messages(topics=['/usb_cam/image_raw/compressed', '/point_cloud']):
    if i.topic == '/usb_cam/image_raw/compressed':
        np_arr = np.fromstring(i.message.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imshow('0', image_np)
        cv2.waitKey(1)
    elif i.topic == '/radar_data':
        mlab.clf(fig)
        gen = pc2.read_points(i.message)

        int_data = list(gen)
        pc = np.asarray(int_data)
        pc = pc[:, :3]
        # pc = StandardScaler().fit_transform(pc)
        # clustering = DBSCAN(eps=0.8, min_samples=20)
        # clustering.fit(pc)
        # label = clustering.labels_
        # pc = filter_cluster(pc, label)
        # print(pc)
        pc = pc[:, :3]
        draw_radar(pc, fig=fig)
        mlab.show(1)

    # input()
