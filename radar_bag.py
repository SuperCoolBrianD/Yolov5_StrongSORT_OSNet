import mayavi.mlab as mlab
import rosbag
from projectutils import draw_radar
from retina_view.msg import MsgRadarPoint
from radar_utils import *
import time
# Press the green button in the gutter to run the script.



bag = rosbag.Bag("record/car_1_ped_1.bag")
topics = bag.get_type_and_topic_info()
# print(topics)
t = 0
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
    if i.topic == '/usb_cam/image_raw/compressed':
        np_arr = np.frombuffer(i.message.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        cv2.imshow('0', image_np)
        cv2.waitKey(1)
    elif i.topic == '/radar_data':
        print((i.message.time_stamp[0] - t)/10**6)
        t = i.message.time_stamp[0]
        # print(type(i.message.points[0]))
        # print(i.message.points[0])
        arr = convert_to_numpy(i.message.points)
        arr = filter_zero(arr)

        if image_np.any():
            img, arr = render_lidar_on_image(arr, image_np, np.pi/2, 0, 0, 0, 0, 0, 9000, 9000)
            print(img.shape)
            draw_radar(arr, fig=fig)

        mlab.clf(fig)
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
        # view = mlab.view()

        # mlab.show(stop=True)

    # input()
