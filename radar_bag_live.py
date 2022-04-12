import sort
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from celluloid import Camera

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
    # if not pc_list:
    #     return [np.zeros([1,5])]
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
    # if pc.shape[0] == 0:
    #     return np.zeros([1,5])
    return pc


def get_bbox(arr):
    x_coord, y_coord, z_coord = arr[:, 0], arr[:, 1], arr[:, 2]
    x_min = min(x_coord)-0.5
    y_min = min(y_coord)-0.5
    x_max = max(x_coord)+0.5
    y_max = max(y_coord)+0.5
    return [x_min, y_min, x_max-x_min, y_max-y_min], np.array([[x_min, y_min, x_max, y_max, 1]])


bag = rosbag.Bag("car_1_ped_1.bag")
# bag = rosbag.Bag("suped_across_1.bag")
topics = bag.get_type_and_topic_info()
# print(topics)

mot_tracker = sort.Sort(min_hits=2, max_age=8, iou_threshold=0.1)

for i in topics[1]:
    print(i)
# print(type(bag))
# fig = mlab.figure(size=(1000, 1000), bgcolor=(0, 0, 0))
radar_d = '/radar_data'
fig, axs = plt.subplots(1, figsize=(6, 6))
bg = bag.read_messages()
s = 0
cv2.imshow('0', np.zeros((480, 640)))
cv2.moveWindow('0', 800, 800)


def animate(g):
    i = next(bg)
    if i.topic == '/usb_cam/image_raw/compressed':
        np_arr = np.frombuffer(i.message.data, np.uint8)

        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # axs[0].imshow(image_np)
        cv2.imshow('0', image_np)
        cv2.waitKey(1)
        #
        # axs[0].imshow(image_np)
    elif i.topic == '/radar_data' or 'radar_data':
        plt.cla()
        axs.set_xlim(-20, 100)
        axs.set_ylim(-20, 100)

        # print(type(i.message.points[0]))
        # print(i.message.points[0])
        arr = convert_to_numpy(i.message.points)
        # axs.scatter(arr[:, 0], arr[:, 1])
        arr = filter_zero(arr)

        pc = arr[:, :4]
        # pc = StandardScaler().fit_transform(pc)
        total_box = np.empty((0, 5))
        if pc.any():
            clustering = DBSCAN(eps=3, min_samples=25)
            clustering.fit(pc)
            label = clustering.labels_
            cls = set_cluster(pc, label)
            for c in cls:
                bbox, box = get_bbox(c)
                if total_box.size == 0:
                    total_box = box
                else:
                    total_box = np.vstack((total_box, box))

                # print(bbox)
                axs.scatter(c[:, 0], c[:, 1])

                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='blue',
                                         facecolor='none')
                axs.add_patch(rect)
            # print(total_box)
            track_bbs_ids = mot_tracker.update(total_box)
            for t in range(track_bbs_ids.shape[0]):
                axs.text(track_bbs_ids[t, 0], track_bbs_ids[t, 1] - 2, str(int(track_bbs_ids[t, -1])), fontsize=22,
                         color='r')
            # print(track_bbs_ids)
            # print('___________________')
            # input()



ani = FuncAnimation(fig, animate, interval=1)
plt.show()
