import mayavi.mlab as mlab
import sys

import cv2

sys.path.append('yolor')
sys.path.append('SVSF_Track')
#from yolor.detect_custom import init_yoloR, detect
from SVSF_Track.MTT_Functions import *
from radar_utils import *
from projectutils import draw_radar
import rosbag
from matplotlib.animation import FuncAnimation
from vis_util import *


# Read recording
bag = rosbag.Bag("record/working.bag")
# bag = rosbag.Bag("record/traffic1.bag")
topics = bag.get_type_and_topic_info()

# old SORT tracker
# mot_tracker = sort.Sort(min_hits=2, max_age=8, iou_threshold=0.1)

for i in topics[1]:
    print(i)

radar_d = '/radar_data'
s = 0
# model, device, colors, names = init_yoloR(weights='yolor/yolor_p6.pt', cfg='yolor/cfg/yolor_p6.cfg',
#                                           names='yolor/data/coco.names', out='inference/output', imgsz=640)
# adjust image visualization
cv2.namedWindow("Camera")
cv2.moveWindow('Camera', 800, 800)
# init loop


# mtx = np.array([[234.45076996, 0., 334.1804498],
#                 [0.,311.6748573,241.50825294],
#                 [0., 0., 1.]])
# mtx = np.array([[1113.5, 0., 974.2446],
#                 [0.,1113.5,586.6797],
#                 [0., 0., 1.]])
mtx = np.array([[747.9932, 0., 655.5036],
                [0., 746.6126, 390.1168],
                [0., 0., 1.]])

def get_img_local(event, x, y, flags, param):
    global nxt
    global img_coord
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        nxt = True
        img_coord = (x, y)


def not_found_button():
    global nxt
    nxt = True

def empty(v):
    pass




cv2.setMouseCallback('Camera', get_img_local)
fig = mlab.figure(size=(1000, 500), bgcolor=(0, 0, 0))
calib_pts = []
cam1 = np.empty((0,0))
v = (-108.20802358222203, 7.280529894768495, 470.76425650815855, ([12.091, -1.047, -2.0325]))

ry = 0
rz = 0.3
tx = 0.1
ty = 0
tz = 0.05
rx = 1.72

cv2.namedWindow('TrackBar')
cv2.resizeWindow('TrackBar', 640, 320)
cv2.createTrackbar('rx', 'TrackBar', 0, 314, empty)
cv2.createTrackbar('ry', 'TrackBar', 0, 314, empty)
cv2.createTrackbar('rz', 'TrackBar', 0, 314, empty)
cv2.createTrackbar('tx', 'TrackBar', 0, 100, empty)
cv2.createTrackbar('ty', 'TrackBar', 0, 100, empty)
cv2.createTrackbar('tz', 'TrackBar', 0, 100, empty)
cv2.setTrackbarPos('rx', 'TrackBar', 158,)
cv2.setTrackbarPos('ry', 'TrackBar', 0,)
cv2.setTrackbarPos('rz', 'TrackBar', 162,)
cv2.setTrackbarPos('tx', 'TrackBar', 57,)
cv2.setTrackbarPos('ty', 'TrackBar', 50,)
cv2.setTrackbarPos('tz', 'TrackBar', 50,)
frame = SRS_data_frame()
for j, i in enumerate(bag.read_messages()):
    sensor = frame.load_data(i)

    # print(idx)
    # print(sensor)

    if frame.full_data:
        # print(abs(abs(frame.camera.message.header.stamp.to_sec() - frame.radar.message.header.stamp.to_sec()) - 1))
        # print(frame.camera.message.header.stamp.to_sec() - frame.radar.message.header.stamp.to_sec())
        print(frame.radar.message.header.stamp.to_sec())
        # print(frame.radar.message.header.stamp.to_sec()- epoch)
        # epoch = frame.radar.message.header.stamp.to_sec()

        image_np = imgmsg_to_cv2(frame.camera.message)
        npts = frame.radar.message.width
        arr_all = pc2_numpy(frame.radar.message, npts)
        print(arr_all.shape)
        # print(np.max(arr_all[:, 0]))
        # print(np.max(arr_all[:, 1]))
        # print(np.max(arr_all[:, 2]))
        # print(np.min(arr_all[:, 0]))
        # print(np.min(arr_all[:, 1]))
        # print(np.min(arr_all[:, 2]))

        arr = filter_zero(arr_all)
        # draw points on plt figure
        pc = arr[:, :4]
        ped_box = np.empty((0, 5))
        total_box, cls = dbscan_cluster(pc, eps=2, min_sample=20)
        if total_box.any() and ped_box.any:
            total_box = np.vstack((total_box, ped_box))
        # yolo detection
        # cam1, detection = detect(source=cam1, model=model, device=device, colors=colors, names=names,
        #                              view_img=False)
        # Radar projection onto camera parameters

        r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)
        new_cam1, cam_arr = render_radar_on_image(arr, cam1, r2c, 9000, 9000)
        rx = cv2.getTrackbarPos('rx', 'TrackBar') / 100
        ry = cv2.getTrackbarPos('ry', 'TrackBar') / 100
        rz = cv2.getTrackbarPos('rz', 'TrackBar') / 100 - 157
        tx = cv2.getTrackbarPos('tx', 'TrackBar') / 10 - 5
        ty = cv2.getTrackbarPos('ty', 'TrackBar') / 10 - 5
        tz = cv2.getTrackbarPos('tz', 'TrackBar') / 10 - 5

        if cls:
            for cc in cls:
                draw_radar(arr_all, fig=fig, pts_scale=0.1, pts_color=(1, 1, 1), view=v)
                bbox = get_bbox_cls(cc)
                bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)
                draw_gt_boxes3d(bbox, fig=fig)
            print('Adjust using trackbar, Press c for next frame')
            while True:
                # new_cam1 = cam1.copy()cccccccccccccc
                rx = cv2.getTrackbarPos('rx', 'TrackBar') / 100
                ry = cv2.getTrackbarPos('ry', 'TrackBar') / 100
                rz = cv2.getTrackbarPos('rz', 'TrackBar') / 100 - 1.57
                tx = cv2.getTrackbarPos('tx', 'TrackBar') / 10 - 5
                ty = cv2.getTrackbarPos('ty', 'TrackBar') / 10 - 5
                tz = cv2.getTrackbarPos('tz', 'TrackBar') / 10 - 5
                r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)

                # extrinsic radar -> pixel coordinate
                # radar -> camera coordinate
                # radar_cam_coord -> rotx(alpha) * radar_cam_coord -> world coordinate with origin at radar (pitch about 5 degree)
                new_cam1, cam_arr = render_radar_on_image(arr, image_np, r2c, 9000, 9000)
                for cc in cls:
                    bbox = get_bbox_cls(cc)
                    bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)
                    bbox = project_to_image(bbox, r2c)
                    draw_projected_box3d(new_cam1, bbox)
                    xyz = np.mean(cc, axis=0).reshape((-1, 1))
                    xyz = xyz[:3, :]
                    cent = project_to_image(xyz, r2c)
                    cent = (int(cent[0, 0]), int(cent[1, 0]))
                    new_cam1 = cv2.circle(new_cam1, cent, 5, (255, 255, 0), thickness=2)

                cv2.imshow('Camera', new_cam1)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    break
            v = mlab.view()
            mlab.clf()
        print(rx, ry, rz, tx, ty, tz)
        print('next frame')

        update = 1
