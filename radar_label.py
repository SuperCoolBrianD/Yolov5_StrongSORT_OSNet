import mayavi.mlab as mlab
import sys
import pandas as pd
import sensor_msgs.point_cloud2 as pc2
import cv2
import pickle
sys.path.append('yolor')
sys.path.append('SVSF_Track')
# from yolor.detect_custom import init_yoloR, detect
from SVSF_Track.MTT_Functions import *
from radar_utils import *
from projectutils import draw_radar
import rosbag
from matplotlib.animation import FuncAnimation



# Read recording
bag = rosbag.Bag("record/calib.bag")

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


mtx = np.array([[234.45076996, 0., 334.1804498],
                [0.,311.6748573,241.50825294],
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

data = pd.read_csv('testfile.csv', header=None)

def pc2_numpy(pc, l):
    arr = np.zeros((l, 5))
    for i ,point in enumerate(pc2.read_points(pc, skip_nans=True)):
        pt_x = point[0]
        pt_y = point[1]
        pt_z = point[2]
        doppler = point[3]
        if pt_x!= 0 and pt_y != 0 and pt_z != 0:
            arr[i, 0] = pt_x
            arr[i, 1] = pt_y
            arr[i, 2] = pt_z
            arr[i, 3] = doppler
        else:
            arr[i, 0] = 0.1
            arr[i, 1] = 0.1
            arr[i, 2] = -100
            arr[i, 3] = doppler
    return arr


# print(data)
print(data[1][0])
all_bb = []
print(pickle.HIGHEST_PROTOCOL)
for j, i in enumerate(bag.read_messages()):
    # print(i.topic)
    # read ros Topic camera or radar
    if j<120:
        continue
    if i.topic == '/image_raw':
        np_arr = np.frombuffer(i.message.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (640, 480), 5)
        # image_np = cv2.remap(image_np, mapx, mapy, cv2.INTER_LINEAR)
        # # crop the image
        # x, y, w, h = roi
        # image_np = image_np[y:y + h, x:x + w]
        cam1 = image_np
    elif i.topic == '/radar_data' or 'radar_data' or '/Radar':
        npts = i.message.width
        arr_all = pc2_numpy(i.message, npts)
        arr = filter_zero(arr_all)
        # draw points on plt figure
        pc = arr[:, :4]
        ped_box = np.empty((0, 5))
        total_box, cls = dbscan_cluster(pc, eps=1, min_sample=20)

        bbframe = []
        if cls:
            for cc in cls:
                bbox = get_bbox_cls_label(cc)
                bbframe.append(bbox)
                print(bbframe)
                # print(bbox)
            update = 1
        all_bb.append(bbframe)
with open(r'test.pkl', 'wb') as d:
    pickle.dump(all_bb, d, 2)
