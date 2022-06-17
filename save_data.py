import mayavi.mlab as mlab
import numpy as np
import rospy
import rosbag
import pickle
from radar_utils import *
# Press the green button in the gutter to run the script.
# bag = rosbag.Bag("record/car.bag")
# topics = bag.get_type_and_topic_info()
# # print(topics)
#
# idx = 0
# for j, i in enumerate(bag.read_messages()):
#
#     if i.topic == '/Radar':
#         npts = i.message.width
#         arr = pc2_numpy(i.message, npts)
#         with open(f'radar_data/{idx}.pkl', "wb") as file:
#             pickle.dump(arr, file)
#         idx+=1

with open('radar_data/1.pkl', 'rb') as f:
    arr_all = pickle.load(f)
    arr = filter_zero(arr_all)


with open('label/1.txt') as file:
    i = 0
    lines = file.readlines()
    x = np.empty([len(lines), 11])
    y = np.empty([len(lines), 5])  # initialize labels
    for i in range(len(lines)):
        line = lines[i].split()
        bbox = get_bbox_coord(float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]))
        pc, _ = extract_pc_in_box3d(arr, bbox.T)
        # x.append(np.std(pc[:, 0]))
        x[i][0] = pc.shape[0]       #num
        x[i][1] = float(line[3])  #L
        x[i][2] = float(line[4])  #W
        x[i][3] = float(line[5])   #S
        x[i][4] = x[i][0]/x[i][1]/x[i][2]/x[i][3]  #density
        x[i][5] = np.std(pc[:, 0]) #stdx
        x[i][6] = np.std(pc[:, 1])  # stdx
        x[i][7] = np.std(pc[:, 2])  # stdz
        x[i][8] = np.std(pc[:, 3])  # stdv
        x[i][9] = pc[:, 3].mean()   # v mean
        x[i][10] = pc[:, 3].max()-pc[:, 3].min() #v range

x = [1, 2, 3, 4, 5]
print(x[1:])


