import mayavi.mlab as mlab
import rospy
import rosbag
import pickle
from radar_utils import *
# Press the green button in the gutter to run the script.
bag = rosbag.Bag("record/car.bag")
topics = bag.get_type_and_topic_info()
# print(topics)

idx = 0
for j, i in enumerate(bag.read_messages()):

    if i.topic == '/Radar':
        npts = i.message.width
        arr = pc2_numpy(i.message, npts)
        with open(f'radar_data/{idx}.pkl', "wb") as file:
            pickle.dump(arr, file)
        idx+=1


