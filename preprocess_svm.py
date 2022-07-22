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
from learn_util import get_features
import pandas as pd
idx = 0
finalx =np.empty([12])
finaly =np.empty([1])

for idx in range(1348):
    print(idx)
    with open(f'radar_data/{idx}.pkl', "rb") as file:
        arr_all = pickle.load(file)
        # arr = filter_zero(arr_all)

    with open(f'dataset/{idx:05d}/ground_truth.txt') as file:
        i = 0
        lines = file.readlines()

    if(not lines):
        continue
    file.close()
    x = np.empty([len(lines), 11]) #initialize inputs
    # y = np.empty([len(lines), 5])  #initialize labels

    for i in range(len(lines)):
        line = lines[i].split()
        if(line[9] == 'car'):
            y = [0]
        elif(line[9] == 'bus'):
            y = [1]
        elif(line[9] == 'person'):
            y = [2]
        elif(line[9] == 'truck'):
            y = [3]
        elif (line[9] == 'no_match'):
            continue
            # y = [4]
        print(line)
        if float(line[0]) == 0 and float(line[1]) == 0 and float(line[2]) == 0:
            continue
        box = [float(i) for i in line[:-1]]
        bbox = get_bbox_coord(float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]),
                              float(line[5]), float(line[6]))
        # print(bbox.T)
        # print(idx)
        pc, _ = extract_pc_in_box3d(arr_all, bbox.T)
        # x.append(np.std(pc[:, 0]))
        if not pc.any():
            continue
        features = get_features(pc, box)

        finalx = np.vstack(([finalx,features]))
        finaly = np.vstack(([finaly, y]))


finalx = finalx[1:]
finaly = finaly[1:]
data = np.hstack((finalx, finaly))
excel = pd.DataFrame(data)
excel.to_csv('data.csv')
# with open(f'labelled_data/x.pkl', "wb") as file:
#     pickle.dump(finalx, file)
#
# with open(f'labelled_data/y.pkl', "wb") as file:
#     pickle.dump(finaly, file)
