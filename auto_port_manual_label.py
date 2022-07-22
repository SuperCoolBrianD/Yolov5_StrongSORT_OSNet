import os

from radar_utils import *
import sys
sys.path.append('yolor')
sys.path.append('SVSF_Track')
from yolor.detect_custom import init_yoloR, detect
import rosbag
from auto_label_util import *
import pickle
import shutil

# Read recording

d = os.listdir('label3')

for i, file in enumerate(d):
    dst = f"dataset/{int(file.strip('.txt')):05d}/ground_truth.txt"
    shutil.copy(f'label3/{file}', dst)
    print(i)


