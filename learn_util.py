import numpy as np


def get_features(pc, bbox):
    n = pc.shape[0]  # num
    l = bbox[3]  # L
    w = bbox[4]  # W
    h = bbox[5]  # S
    d = n / l / w / h  # density
    stdx = np.std(pc[:, 0])  # stdx
    stdy = np.std(pc[:, 1])  # stdx
    stdz = np.std(pc[:, 2])  # stdz
    stdv = np.std(pc[:, 3])  # stdv
    vmean = pc[:, 3].mean()  # v mean
    vrange = pc[:, 3].max() - pc[:, 3].min()  # v range

    return n, l, w, h, d, stdx, stdy, stdz, stdv, vmean, vrange
