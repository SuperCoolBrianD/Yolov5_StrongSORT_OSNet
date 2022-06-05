import mayavi.mlab as mlab
import sys

import cv2

sys.path.append('yolor')
sys.path.append('SVSF_Track')
from yolor.detect_custom import init_yoloR, detect
from SVSF_Track.MTT_Functions import *
from radar_utils import *
from projectutils import draw_radar
import rosbag
from matplotlib.animation import FuncAnimation



# Read recording
bag = rosbag.Bag("record/synch_1.bag")
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
mtx = np.array([[1113.5, 0., 974.2446],
                [0.,1113.5,586.6797],
                [0., 0., 1.]])
dist = np.array([[-0.06624252, -0.00059409, -0.00183169,  0.0030411,   0.00063524]])

h,  w = 480, 640
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
img_coord = 0
nxt = False


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

cv2.setMouseCallback('Camera', get_img_local)
fig = mlab.figure(size=(1000, 500), bgcolor=(0, 0, 0))
calib_pts = []
cam1 = np.empty((0,0))
v = (-108.20802358222203, 7.280529894768495, 470.76425650815855, ([12.091, -1.047, -2.0325]))
for j, i in enumerate(bag.read_messages()):
    # read ros Topic camera or radar
    if i.topic == '/usb_cam/image_raw/compressed':
        np_arr = np.frombuffer(i.message.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (640, 480), 5)
        # image_np = cv2.remap(image_np, mapx, mapy, cv2.INTER_LINEAR)
        # # crop the image
        # x, y, w, h = roi
        # image_np = image_np[y:y + h, x:x + w]
        cam1 = image_np
    elif i.topic == '/Camera':
        np_arr = np.frombuffer(i.message.data, np.uint8)
        # image_np = cv2.imdecode(np_arr, cv2.IMREAD_)
        image_np = imgmsg_to_cv2(i.message)
        cam1 = image_np
    elif i.topic == '/radar_data' or i.topic== '/Radar':
        if i.topic == '/Radar':
            npts = i.message.width
            arr_all = pc2_numpy(i.message, npts)
            tm = i.message.header.stamp.to_sec()

        else:
            tm = i.message.time_stamp[0]/10**6
            # convert SRS message to Numpy
            arr_all = convert_to_numpy(i.message.points)
            # Filter zero doppler points
        arr = filter_zero(arr_all)
        # draw points on plt figure
        plt.cla()
        pc = arr[:, :4]
        ped_box = np.empty((0, 5))
        total_box, cls = dbscan_cluster(pc, eps=2, min_sample=20)
        if total_box.any() and ped_box.any:
            total_box = np.vstack((total_box, ped_box))
        if cam1.any():
            # yolo detection
            # cam1, detection = detect(source=cam1, model=model, device=device, colors=colors, names=names,
            #                              view_img=False)
            # Radar projection onto camera parameters
            ry = 0
            rz = 0.3
            tx = 0.1
            ty = 0
            tz = 0.05
            rx = 1.72
            r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)
            cam1, cam_arr = render_radar_on_image(arr, cam1, r2c, 9000, 9000)
            if cls:
                for cc in cls:
                    bbox = get_bbox_cls(cc)
                    bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)
                    bbox = project_to_image(bbox, r2c)
                    draw_projected_box3d(cam1, bbox)
                for cc in cls:
                    # draw_radar(cc, fig=fig, pts_color=(0, 0, 1),
                    #            pts_scale=2, view=v)
                    # for ccc in cls:
                    #     draw_radar(ccc, fig=fig, pts_color=(1, 0, 0),
                    #                pts_scale=1, view=v)
                    draw_radar(arr_all, fig=fig, pts_scale=0.1, pts_color=(1, 1, 1), view=v)
                    bbox = get_bbox_cls(cc)
                    bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)
                    draw_gt_boxes3d(bbox, fig=fig)
                    record = True
                    print('Click on the object in camera or Press c to skip to next object')
                    xyz = np.mean(cc, axis=0).reshape((-1, 1))
                    print(f'Current object doppler:{xyz[-1, 0]}')
                    xyz = xyz[:3, :]

                    new_cam1 = cam1.copy()

                    cent = project_to_image(xyz, r2c)
                    cent = (int(cent[0, 0]), int(cent[1, 0]))
                    new_cam1 = cv2.circle(new_cam1, cent, 5, (255, 255, 0), thickness=2)
                    while nxt is False:
                        cv2.imshow('Camera', new_cam1)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('c'):
                            print('Cannot found in image')
                            record =False
                            nxt =True

                    if record is True:
                        calib_pts.append((xyz.flatten(), img_coord))
                    print(calib_pts)
                    nxt = False
                    v = mlab.view()
                    mlab.clf()
            print('next frame')
            update = 1
