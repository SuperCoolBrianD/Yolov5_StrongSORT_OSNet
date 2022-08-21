from radar_utils import *
from auto_label_util import IOU

img = [612, 242, 681, 296]
radar = [[ 578,  220,  590,  222],
    [1167,  140, 1185,  172],
    [ 631,  275,  658,  290]]

for i in radar:
    iou = IOU(i, img)
    print(iou)