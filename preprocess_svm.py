import numpy as np

from radar_utils import *
import sys
sys.path.append('yolor')
sys.path.append('SVSF_Track')
#from yolor.detect_custom import init_yoloR, detect
import rosbag
from auto_label_util import *
from projectutils import draw_radar
from vis_util import *
import pickle


# # Read recording
# bag = rosbag.Bag("record/car.bag")
#
# topics = bag.get_type_and_topic_info()
#
# half = False
# radar_d = '/radar_data'
# s = 0
# # model, device, colors, names = init_yoloR(weights='yolor/yolor_p6.pt', cfg='yolor/cfg/yolor_p6.cfg',
# #                                           names='yolor/data/coco.names', out='inference/output', imgsz=1280, half=False)
#
#
# mtx = np.array([[770, 0., 639.5],
#                 [0.,770,381.6629],
#                 [0., 0., 1.]])
#
# calib_pts = []
# cam1 = np.empty((0,0))
# v = (-108.20802358222203, 7.280529894768495, 470.76425650815855, ([12.091, -1.047, -2.0325]))
#
#
# def convert_xyxy(box):
#     return (box[0], box[1]), (box[2], box[3])
#
#
# def convert_topy_bottomx(box):
#     topx = min(box[0], box[2])
#     topy = min(box[1], box[3])
#     bottomx = max(box[0], box[2])
#     bottomy = max(box[1], box[3])
#     return [topx, topy, bottomx, bottomy]
#
#
#
# # print(data)
# all_bb = []
# idx = 0
# missidx = 0
#
# rx = 1.56
# ry = 0
# rz = -.02
# tx = 0
# ty = 0
# tz = 0.1
# fig = mlab.figure(size=(1000, 500), bgcolor=(0, 0, 0))
# r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)
# frame = SRS_data_frame()
# epoch = None
# cnt = 0
# classes = ['person', 'car', 'truck', 'cyclist', 'bus', 'no_match', 'outside_fov']
# v = (-108.20802358222203, 7.280529894768495, 470.76425650815855, ([12.091, -1.047, -2.0325]))
# for j, i in enumerate(bag.read_messages()):
#     print(j)
#     if j <= cnt:
#         continue
#     # read ros Topic camera or radar
#     sensor = frame.load_data(i)
#     # print(idx)
#     # print(sensor)
#     if sensor == '/Radar':
#         file = open(f'label/{idx}.txt', 'w')
#         idx+=1
#     if frame.full_data:
#         print(abs(abs(frame.camera.message.header.stamp.to_sec()- frame.radar.message.header.stamp.to_sec())-1))
#         # print(frame.camera.message.header.stamp.to_sec()- epoch)
#         image_np = imgmsg_to_cv2(frame.camera.message)
#         npts = frame.radar.message.width
#         arr_all = pc2_numpy(frame.radar.message, npts)
#         # draw points on plt figure
#         arr = filter_zero(arr_all)
#
#         total_box, cls = dbscan_cluster(arr, eps=2, min_sample=20)
#         # image_np_detection, detection = detect(source=image_np, model=model, device=device, colors=colors, names=names,
#         #                          view_img=False, half=half)
#
#         # image_np_detection, cam_arr = render_radar_on_image(arr_all, image_np_detection, r2c, 9000, 9000)
#         # with open(f'label/{idx}.txt', 'w') as file:
#         # file.write(f'{frame.radar.message.header.stamp.to_sec()}')
#         # file.write('\n')
#         image_np_detection = image_np.copy()
#         # cv2.imshow('detection', image_np_detection)
#         # cv2.waitKey(0)
#
#             with open(f'radar_data/{idx}.pkl', "rb") as file:
#             for cc in cls:
#                 # draw_radar(arr_all, fig=fig, pts_scale=0.1, pts_color=(1, 1, 1), view=v)
#                 image_np_detection, cam_arr = render_radar_on_image(cc, image_np_detection, r2c, 9000, 9000)
#                 bbox = get_bbox_cls(cc)
#                 # print(bbox)
#                 # xyzlwhrz -> Nx3 corner
#                 bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)
#
#                 # bbox is a convex hull
#                 pc, _ = extract_pc_in_box3d(arr_all, bbox.T)
#                 # feature extraction
#                 stdx = np.std(pc[:, 0])
#                 stdy = np.std(pc[:, 1])
#                 stdz = np.std(pc[:, 2])
#                 stdv = np.std(pc[:, 3])
#                 (x-> [x,y,z,stdx, stdy, stdz.....], y->)
#                 # draw_gt_boxes3d(bbox, fig=fig)
#                 # draw_radar(pc, fig=fig, pts_scale=0.1, pts_color=(1, 1, 1), view=v)
#                 # mlab.show()
#                 # bbox = project_to_image(bbox, r2c)
#                 # pts = project_to_image(cc.T, r2c)
#                 # # print(pts)
#                 # box2d = get_bbox_2d(pts.T)
#                 # draw_projected_box3d(image_np_detection, bbox)
#                 # cv2.rectangle(image_np_detection, box2d[0], box2d[1], (255, 255, 0))
#                 # box2d = [box2d[0][0], box2d[0][1], box2d[1][0], box2d[1][1]]
#                 # box2d = convert_topy_bottomx(box2d)
#
#                 input()
#                 # cv2.imshow('no detection', image_np)
#
#                 mlab.clf()
#         # idx+=1
#         mlab.clf()
#         frame.clear_data()
#         file.close()
idx = 0
finalx =np.empty([11])
finaly =np.empty([5])

for idx in range(1348):
    with open(f'radar_data/{idx}.pkl', "rb") as file:
        arr_all = pickle.load(file)
        arr = filter_zero(arr_all)

    with open(f'label/{idx}.txt') as file:
        i = 0
        lines = file.readlines()

    if(not lines):
        continue
    file.close()
    x = np.empty([len(lines), 11]) #initialize inputs
    y = np.empty([len(lines), 5])  #initialize labels

    for i in range(len(lines)):
        line = lines[i].split()
        bbox = get_bbox_coord(float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]),
                              float(line[5]), float(line[6]))
        pc, _ = extract_pc_in_box3d(arr, bbox.T)
        # x.append(np.std(pc[:, 0]))
        x[i][0] = pc.shape[0]  # num
        x[i][1] = float(line[3])  # L
        x[i][2] = float(line[4])  # W
        x[i][3] = float(line[5])  # S
        x[i][4] = x[i][0] / x[i][1] / x[i][2] / x[i][3]  # density
        x[i][5] = np.std(pc[:, 0])  # stdx
        x[i][6] = np.std(pc[:, 1])  # stdx
        x[i][7] = np.std(pc[:, 2])  # stdz
        x[i][8] = np.std(pc[:, 3])  # stdv
        x[i][9] = pc[:, 3].mean()  # v mean
        x[i][10] = pc[:, 3].max() - pc[:, 3].min()  # v range
        finalx = np.vstack(([finalx,x[i]]))

        #label to one hot vector
        if(line[9] == 'car'):
            y[i] = [1, 0, 0, 0, 0]
        elif(line[9] == 'bus'):
            y[i] = [0, 1, 0, 0, 0]
        elif(line[9] == 'person'):
            y[i] = [0, 0, 1, 0, 0]
        elif(line[9] == 'truck'):
            y[i] = [0, 0, 0, 1, 0]
        elif (line[9] == 'no_match'):
            y[i] = [0, 0, 0, 0, 1]
        finaly = np.vstack(([finaly, y[i]]))

finalx = finalx[1:]
finaly = finaly[1:]











