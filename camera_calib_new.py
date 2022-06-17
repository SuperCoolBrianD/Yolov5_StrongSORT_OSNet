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


bag = rosbag.Bag("record/car_calib.bag")
topics = bag.get_type_and_topic_info()

x = 0

for j, i in enumerate(bag.read_messages()):
    # read ros Topic camera or radar
    if i.topic == '/usb_cam/image_raw/compressed':
        np_arr = np.frombuffer(i.message.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # image_np = cv2.resize(image_np, (1280, 720))
        cv2.imwrite('cam_calib2/' + str(x) + '.jpg', image_np)
        x = x + 1
    elif i.topic == '/Camera':
        np_arr = np.frombuffer(i.message.data, np.uint8)
        # image_np = cv2.imdecode(np_arr, cv2.IMREAD_)
        image_np = imgmsg_to_cv2(i.message)
        # image_np = cv2.resize(image_np, (1280, 720))
        cv2.imwrite('cam_calib2/' + str(x) + '.jpg', image_np)
        x = x + 1



