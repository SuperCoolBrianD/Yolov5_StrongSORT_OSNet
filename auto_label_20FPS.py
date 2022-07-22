import os

from radar_utils import *
import sys
sys.path.append('yolor')
sys.path.append('SVSF_Track')
from yolor.detect_custom import init_yoloR, detect
import rosbag
from auto_label_util import *
import pickle


# Read recording



half = False
radar_d = '/radar_data'
s = 0
model, device, colors, names = init_yoloR(weights='yolor/yolor_p6.pt', cfg='yolor/cfg/yolor_p6.cfg',
                                          names='yolor/data/coco.names', out='inference/output', imgsz=1280, half=False)

# adjust image visualization
cv2.namedWindow("Camera")
cv2.moveWindow('Camera', 800, 800)
# init loop


mtx = np.array([[770, 0., 639.5],
                [0.,770,381.6629],
                [0., 0., 1.]])

calib_pts = []
cam1 = np.empty((0,0))
v = (-108.20802358222203, 7.280529894768495, 470.76425650815855, ([12.091, -1.047, -2.0325]))




bag = rosbag.Bag("record/tripod.bag")

# print(data)
all_bb = []
r_idx = 0
missidx = 0
c = 0
r = 0
# for tripod.bag
all_idx = 0
ct = 0
rx, ry, rz, tx, ty, tz = 1.58, 0.0, 0.040000000000000036, 0.7000000000000002, 1.0999999999999996, 0.0
r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)
frame = SRS_data_frame()
epoch = None
t = 0
for j, i in enumerate(bag.read_messages()):
    if i.topic == '/Camera':
        print(i.message.header.stamp.to_sec() - ct)
        c+=1
        ct = i.message.header.stamp.to_sec()
    else:
        # print(i.message.header.stamp.to_sec()-t)
        r+=1
        t = i.message.header.stamp.to_sec()
    # read ros Topic camera or radar
    sensor = frame.load_data(i)
    # print(idx)
    # print(sensor)
    if sensor == '/Radar':
        # file = open(f'dataset/{idx:05d}.txt', 'w')
        r_idx+=1
    if frame.full_data:
        # file = open(f'dataset/{idx:05d}.txt', 'w')
        all_idx+=1
        # print(abs(abs(frame.camera.message.header.stamp.to_sec()- frame.radar.message.header.stamp.to_sec())-1))
        # # print(frame.camera.message.header.stamp.to_sec()- epoch)
        # image_np = imgmsg_to_cv2(frame.camera.message)
        # cv2.imwrite(f'dataset/{idx-1:05d}/camera.png', image_np)
        # npts = frame.radar.message.width
        # arr_all = pc2_numpy(frame.radar.message, npts)
        # pickle.dump(arr_all, open(f"dataset/{idx-1:05d}/radar.pkl", 'wb'))
        # # draw points on plt figure
        # arr = filter_zero(arr_all)
        # total_box, cls = dbscan_cluster(arr, eps=2, min_sample=20)
        # image_np_detection, detection = detect(source=image_np, model=model, device=device, colors=colors, names=names,
        #                          view_img=False, half=half)
        #
        # image_np_detection, cam_arr = render_radar_on_image(arr_all, image_np_detection, r2c, 9000, 9000)
        # if cls:
        #     for cc in cls:
        #         bbox = get_bbox_cls(cc)
        #         # print(bbox)
        #         bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)
        #
        #         bbox = project_to_image(bbox, r2c)
        #         pts = project_to_image(cc.T, r2c)
        #         # print(pts)
        #         box2d = get_bbox_2d(pts.T)
        #         draw_projected_box3d(image_np_detection, bbox)
        #         cv2.rectangle(image_np_detection, box2d[0], box2d[1], (255, 255, 0))
        #         cv2.rectangle(image_np_detection, box2d[0], box2d[1], (255, 255, 0))
        #         box2d = [box2d[0][0], box2d[0][1], box2d[1][0], box2d[1][1]]
        #         box2d = convert_topy_bottomx(box2d)
        #         matched = find_gt(box2d, detection)
        #         label = get_bbox_cls_label(cc, matched[0][1])
        #         # bbframe.append(bbox)
        #         file.write(' '.join([str(num) for num in label]))
        #         file.write('\n')
        #         # if matched[0]:
        #         #     mbox2s = convert_xyxy(matched[0][0])
        #         #     cv2.rectangle(image_np, mbox2s[0], mbox2s[1], (255, 255, 0))
        #         #     cv2.imshow('no detection', image_np)
        #         #     cv2.waitKey(100)
        # # cv2.imshow('detection', image_np_detection)
        # # cv2.imshow('no detection', image_np)
        # # cv2.waitKey(1)
        # # idx+=1
        # # frame.clear_data()
        # idx+=1
        # file.close()


print(c)
print(r)
print(missidx)