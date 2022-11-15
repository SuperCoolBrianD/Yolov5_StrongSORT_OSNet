import imp
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.point_cloud2 import _get_struct_fmt
from sensor_msgs.msg import Image
import sys
import math
import matplotlib.path as mpltPath
from sensor_msgs.msg import PointCloud2, PointField
import struct
import yaml
from yaml import Loader

import pickle
import os
import torch
import random

# from Yolov5_StrongSORT_OSNet.strong_sort.deep.reid.torchreid.metrics import compute_distance_matrix

# from Yolov5_StrongSORT_OSNet.track_custom import load_weight_sort, process_track
_DATATYPES = {}
_DATATYPES[PointField.INT8] = ('b', 1)
_DATATYPES[PointField.UINT8] = ('B', 1)
_DATATYPES[PointField.INT16] = ('h', 2)
_DATATYPES[PointField.UINT16] = ('H', 2)
_DATATYPES[PointField.INT32] = ('i', 4)
_DATATYPES[PointField.UINT32] = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)


def cfg_read(filename):
    read_file = open(filename, 'r')
    data = yaml.load(read_file, Loader=Loader)
    r2c_ext = data['radar_to_cam_extrinsic']['matrix']
    c2g_ext = data['cam_to_WCS_extrinsic']['matrix']
    intrinsic = data['cam_intrinsic']['matrix']
    road_mask_x = data['road_mask']['array']['x']
    road_mask_y = data['road_mask']['array']['y']
    intrinsic = np.asarray(intrinsic).reshape(3, 3)
    road_mask = np.vstack((road_mask_x, road_mask_y)).T
    read_file.close()
    return r2c_ext, c2g_ext, intrinsic, road_mask


def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]
    pts = points[:3, :]
    # Change to homogenous coordinate
    pts = np.vstack((pts, np.ones((1, num_pts))))

    pts = proj_mat @ pts
    pts[:2, :] /= pts[2, :]
    points = np.vstack((pts[:2, :], points[3:, :]))
    return points


def transform_radar(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]
    pts = points[:3, :]
    # Change to homogenous coordinate
    pts = np.vstack((pts, np.ones((1, num_pts))))
    pts = proj_mat @ pts
    points = np.vstack((pts[:3, :], points[3:, :]))
    return points


def roty(t):
    """
    Rotation about the y-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotx(t):
    """
    Rotation about the y-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def rotz(t):
    """
    Rotation about the y-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def extrinsic_matrix(rx, ry, rz, tx, ty, tz):
    rx = rotx(rx)
    ry = roty(ry)
    rz = rotz(rz)
    t = np.eye(4)
    t[0, 3] = tx
    t[1, 3] = ty
    t[2, 3] = tz
    rr = rx @ ry @ rz
    r = np.eye(4)
    r[:3, :3] = rr
    return r @ t


def c_extrinsic_matrix(rx, ry, rz, tx, ty, tz):
    rx = rotx(rx)
    ry = roty(ry)
    rz = rotz(rz)
    R = rx @ ry @ rz
    C = np.array([[tx, ty, tz]]).T
    t = -R @ C
    mtr = np.eye(4)
    mtr[:3, :3] = R.T
    mtr[:3, 2] = t
    return mtr


def cam_radar(rx, ry, rz, tx, ty, tz, c):
    cam_matrix = np.eye(4)
    cam_matrix[:3, :3] = c
    extrinsic = extrinsic_matrix(rx, ry, rz, tx, ty, tz)

    proj_radar2cam = cam_matrix @ extrinsic

    return proj_radar2cam


def camera_pose(rx, ry, rz, tx, ty, tz, c):
    cam_matrix = np.eye(4)
    cam_matrix[:3, :3] = c
    extrinsic = extrinsic_matrix(rx, ry, rz, tx, ty, tz)
    # print(radar_matrix)
    proj_radar2cam = cam_matrix @ extrinsic
    # print(cam_matrix)
    return proj_radar2cam


def render_radar_on_image(pts_radar, img, proj_radar2cam, img_width, img_height):
    """functions to project radar points on image"""
    # projection matrix (project from velo2cam2)
    img = img.copy()
    # apply projection
    pts_2d = project_to_image(pts_radar.transpose(), proj_radar2cam)
    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0)
                    )[0]
    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]
    # Retrieve depth from radar
    imgfov_pc_velo = pts_radar[:, :3][inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_radar2cam @ imgfov_pc_velo.transpose()

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        cl = int(1200 / depth)
        if cl > 255:
            cl = 255
        elif cl < 0:
            cl = 0

        color = cmap[cl, :]
        cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                         int(np.round(imgfov_pc_pixel[1, i]))),
                   2, color=color, thickness=-1)

    return img, imgfov_pc_pixel


def filter_cluster(pc, label):
    msk = label != -1
    pc = pc[msk, :]
    if pc.shape[0] == 0:
        return np.zeros([1, 5])
    return pc


def set_cluster(pc, label):
    l = np.unique(label)
    pc_list = []
    for i in l:
        if i != -1:
            msk = label == i
            pts = pc[msk, :]
            pc_list.append(pts)
    return pc_list


def convert_to_numpy(pc):
    l = len(pc)
    arr = np.zeros((l, 5))
    for i, point in enumerate(pc):
        if point.x != 0 and point.y != 0 and point.z != 0:
            arr[i, 0] = point.x
            arr[i, 1] = point.y
            arr[i, 2] = point.z
            arr[i, 3] = point.doppler
        else:
            arr[i, 0] = 0.1
            arr[i, 1] = 0.1
            arr[i, 2] = -100
            arr[i, 3] = point.doppler
    return arr


def filter_zero(pc, thresh=0.05):
    """Filter low velocity points"""
    mask = np.abs(pc[:, 3]) > thresh
    pc = pc[mask, :]
    return pc


def filter_move(pc):
    """Filter low velocity points"""
    mask = np.abs(pc[:, 3]) < 0.05
    pc = pc[mask, :]
    return pc


def get_bbox(arr):
    x_coord, y_coord, z_coord = arr[:, 0], arr[:, 1], arr[:, 2]
    x_min = min(x_coord) - 1
    z_min = min(z_coord) - 1
    x_max = max(x_coord) + 1
    z_max = max(z_coord) + 1
    return [x_min, z_min, x_max - x_min, z_max - z_min], np.array([[x_min, z_min, x_max, z_max, 1]])


def get_bbox_2d(arr):
    x_coord, y_coord, z_coord = arr[:, 0], arr[:, 1], arr[:, 2]
    x_min = min(x_coord) - 1
    y_min = min(y_coord) - 1
    x_max = max(x_coord) + 1
    y_max = max(y_coord) + 1
    return (int(x_min), int(y_max)), (int(x_max), int(y_min))


def get_bbox_cls(arr):
    x_coord, y_coord, z_coord = arr[:, 0], arr[:, 1], arr[:, 2]
    x_min = min(x_coord)
    x_max = max(x_coord)
    y_min = min(y_coord)
    y_max = max(y_coord)
    z_min = min(z_coord)
    z_max = max(z_coord)
    return np.array([x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2, z_min + (z_max - z_min) / 2,
                     x_max - x_min, y_max - y_min, z_max - z_min, 0])


def get_bbox_cls_label(arr, clf):
    x_coord, y_coord, z_coord = arr[:, 0], arr[:, 1], arr[:, 2]
    x_min = min(x_coord)
    x_max = max(x_coord)
    y_min = min(y_coord)
    y_max = max(y_coord)
    z_min = min(z_coord)
    z_max = max(z_coord)
    return [x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2, z_min + (z_max - z_min) / 2,
            x_max - x_min, y_max - y_min, z_max - z_min, 0, 0, 0, clf]


def get_bbox_cls_label_kitti(arr, clf, box2d):
    x_coord, y_coord, z_coord = arr[:, 0], arr[:, 1], arr[:, 2]
    x_min = min(x_coord)
    x_max = max(x_coord)
    y_min = min(y_coord)
    y_max = max(y_coord)
    z_min = min(z_coord)
    z_max = max(z_coord)
    return [clf, 0, 0, 0, box2d[0], box2d[1], box2d[2], box2d[3],
            x_max - x_min, y_max - y_min, z_max - z_min, x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2,
            z_min + (z_max - z_min) / 2, 0, 0]


def get_bbox_coord(t1, t2, t3, w, h, l, rz, is_homogenous=False):
    # 3d bounding box dimensions

    # 3D bounding box vertices [3, 8]
    z = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]
    x = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    box_coord = np.vstack([x, y, z])
    R = rotz(rz)  # [3, 3]
    points_3d = R @ box_coord
    points_3d[0, :] = points_3d[0, :] + t1
    points_3d[1, :] = points_3d[1, :] + t2
    points_3d[2, :] = points_3d[2, :] + t3
    if is_homogenous:
        points_3d = np.vstack((points_3d, np.ones(points_3d.shape[1])))

    return points_3d


def Cam2Ground(px, py, P):
    p = np.array([[px, py, 1]]).T  # pixel coordinate
    W = P @ p  # world coordinate
    X = W[0, 0] / W[2, 0]
    Z = W[1, 0] / W[2, 0]
    return X, Z


def Cam2WCS(rx, ry, rz, tx, ty, tz, intrinsic):
    rx = rotx(rx)
    ry = roty(ry)
    rz = rotz(rz)
    rr = rotx(-0.5 * np.pi) @ roty(0.5 * np.pi) @ rx @ ry @ rz
    # rotation matrix
    rr[0, 0] = rr[1, 1] = rr[1, 2] = rr[2, 0] = 0
    rr[1, 0] = -1
    rr = rr.T
    c = np.array([[tx, ty, tz]]).T
    t = -rr @ c  # translation vector
    P = intrinsic @ np.array([[rr[0, 0], rr[0, 1], t[0, 0]],  # extrinsic matrix
                              [rr[1, 0], rr[1, 1], t[1, 0]],
                              [rr[2, 0], rr[2, 1], t[2, 0]]])
    P = np.linalg.inv(P)  # world coordinate
    return P


class calib():
    def __init__(self, rx, ry, rz, tx, ty, tz, mtx, alpha, height):
        self.mtx_p = np.eye(4)
        self.mtx_p[:3, :3] = mtx
        self.r2c_e = extrinsic_matrix(rx, ry, rz, tx, ty, tz)
        self.c2g = extrinsic_matrix(alpha, 0, 0, 0, height, 0)
        self.g2c_p = cam_radar(-alpha, 0, 0, 0, -height, 0, mtx)
        self.c2wcs = Cam2WCS(alpha, 0, 0, 0, 0, -height, mtx)


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=1):
    qs = qs.astype(np.int32).transpose()
    for k in range(0, 4):
        # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)


def dbscan_cluster(pc, eps=3, min_sample=25, axs=None):
    """
    clustering algorithm
    pc: radar point cloud (Nx5)
    eps: dbscan parameter for closeness to be considered as neighbour
    min_sample: minimum sample to be considered as neighbourhood
    """

    total_box = np.empty((0, 5))
    # if empty return None
    if not pc.any():
        return total_box, None
    # Init DBSCAN algorithm
    clustering = DBSCAN(eps=eps, min_samples=min_sample)
    # cluster pc
    mpc = pc[:, :4]
    # mpc = StandardScaler().fit_transform(mpc)
    clustering.fit(mpc)

    # generate cluster label for each point
    label = clustering.labels_
    # filter cluster by label from DBSCAN
    cls = set_cluster(pc, label)
    for i, c in enumerate(cls):
        # get 2D bbox of cluster
        bbox, box = get_bbox(c)
        box[0, -1] = i
        if total_box.size == 0:
            total_box = box
        else:
            total_box = np.vstack((total_box, box))
    return total_box, cls


def plot_box(bbox, c, axs):
    # axs.scatter(c[:, 0], c[:, 1], s=0.5)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='blue',
                             facecolor='none')
    axs.add_patch(rect)


def get_centroid(bbox):
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def match_measurement(detection_list, tracks):
    distance = 999
    picked = None
    thresh = 1
    for index, det in enumerate(detection_list):
        if det.sensor != 'Camera' and det.sensor != 'Radar_track':
            d = np.sqrt((det.centroid[0] - tracks[0]) ** 2 + (det.centroid[2] - tracks[1]) ** 2)
            if d < distance and d < thresh:
                distance = d
                picked = index
    return picked


def match_track_camera(camera_list, centroid):
    for cam in camera_list:
        bbox = cam[0][:4]
        if centroid[0] > bbox[0] and centroid[0] < bbox[2] and centroid[1] > bbox[1] and centroid[1] < bbox[3]:
            return cam[2]


class radar_object:
    """radar object for data abstraction"""

    def __init__(self, current_p):
        """current_p: first position where the track is initialized Tuple ((x, y), time)"""
        self.tracklet = [current_p]
        self.speed = 'NaN'
        self.current_p = current_p
        self.life = 10

    def upd(self, current_p):
        """update tracklet"""
        self.life = 10
        self.tracklet.append(current_p)

    def update_speed(self):
        """calculate speed based on moving average"""
        speed_hist = []
        l = len(self.tracklet)
        average_window = 6
        if l > 5:
            for i in range(2, average_window + 1):
                speed_hist.append(np.sqrt(((self.tracklet[-1][0][0] - self.tracklet[-i][0][0]) ** 2 +
                                           (self.tracklet[-1][0][1] - self.tracklet[-i][0][1]) ** 2)) / (
                                          self.tracklet[-1][1] - self.tracklet[-i][1]) * 3.6)
            self.speed = sum(speed_hist) / len(speed_hist)
        else:
            self.speed = 'Measuring'


# class SRS_data_frame:
#     def __init__(self):
#         self.camera = None
#         self.radar = None
#         self.has_radar = False
#         self.has_camera = False
#         self.full_data = False
#         self.radar_frame = 0
#         self.camera_frame = 0

#     def load_data(self, data):
#         if self.full_data:
#             self.clear_data()
#         if isinstance(data, PointCloud2):
#             self.radar = data
#             self.has_radar = True
#             self.radar_frame +=1
#             sensor = '/Radar'
#         elif isinstance(data, Image):
#             self.camera = data
#             self.has_camera = True
#             self.camera_frame+=1
#             sensor = '/Camera'
#         if self.has_radar and self.has_camera:
#             self.full_data = True
#         return sensor

#     def clear_data(self):
#         self.camera = None
#         self.radar = None
#         self.has_radar = False
#         self.has_camera = False
#         self.full_data = False


# class SRS_data_frame:
#     def __init__(self):
#         self.camera = None
#         self.radar = None
#         self.has_radar = False
#         self.has_camera = False
#         self.full_data = False
#         self.radar_frame = 0
#         self.camera_frame = 0

#     def load_data(self, data):
#         if self.full_data:
#             self.clear_data()
#         if isinstance(data, PointCloud2):
#             self.radar = data
#             self.has_radar = True
#             self.radar_frame +=1
#             sensor = '/Radar'
#         elif isinstance(data, Image):
#             self.camera = data
#             self.has_camera = True
#             self.camera_frame+=1
#             sensor = '/Camera'
#         if self.has_radar and self.has_camera:
#             self.full_data = True
#         return sensor

#     def clear_data(self):
#         self.camera = None
#         self.radar = None
#         self.has_radar = False
#         self.has_camera = False
#         self.full_data = False


class SRS_data_frame:
    def __init__(self):
        self.camera = None
        self.radar = None
        self.has_radar = False
        self.has_camera = False
        self.full_data = False
        self.radar_frame = 0
        self.camera_frame = 0

    def load_data(self, data):
        if self.full_data:
            self.clear_data()
        if data.topic == '/Radar':
            self.radar = data
            self.has_radar = True
            self.radar_frame += 1
        elif data.topic == '/Camera':
            self.camera = data
            self.has_camera = True
            self.camera_frame += 1
        if self.has_radar and self.has_camera:
            self.full_data = True
        return data.topic

    def load_data_radar(self, data):
        self.radar = [data]
        self.has_radar = True
        if self.has_radar and self.has_camera:
            self.full_data = True

    def load_data_camera(self, data):
        self.camera = data
        self.has_camera = True
        if self.has_radar and self.has_camera:
            self.full_data = True

    def clear_data(self):
        self.camera = None
        self.radar = None
        self.has_radar = False
        self.has_camera = False
        self.full_data = False


class SRS_data_frame_buffer:
    def __init__(self):
        self.camera = []
        self.radar = []
        self.has_radar = False
        self.has_camera = False
        self.full_data = False

    def load_data_radar(self, data):
        self.radar.append(data)
        self.has_radar = True
        if self.has_radar and self.has_camera:
            self.full_data = True

    def load_data_camera(self, data):
        self.camera.append(data)
        self.has_camera = True
        if self.has_radar and self.has_camera:
            self.full_data = True

    def clear(self):
        self.camera = []
        self.radar = []
        self.has_radar = False
        self.has_camera = False
        self.full_data = False


class SRS_data_frame_buffer_sort:
    def __init__(self):
        self.camera = []
        self.radar = []
        self.has_radar = False
        self.has_camera = False
        self.full_data = False

    def load_data(self, data):
        if data[1] == 'c':
            self.camera.append(data[0])
            self.has_camera = True
            if self.has_radar and self.has_camera:
                self.full_data = True
        else:
            self.radar.append(data[0])
            self.has_radar = True
            if self.has_radar and self.has_camera:
                self.full_data = True

    def clear(self):
        self.camera = []
        self.radar = []
        self.has_radar = False
        self.has_camera = False
        self.full_data = False


class DetectedObject:
    def __init__(self, r_d=None, c_d=None, trk=None):
        if c_d and not r_d:
            self.cam_label = c_d[1]
            self.cam_box = c_d[0]
            self.cam_id = c_d[2]
            self.sensor = "Camera"
            self.cls = None
            self.cam_rad_iou = None
            self.rad_label = None
            self.rad_box = None
            self.rad_box2d = None
            self.centroid = None
            self.rad_id = None
        elif r_d and not c_d:
            self.cls = r_d[0]
            self.centroid = r_d[1]
            self.rad_box = r_d[2]
            self.rad_box2d = r_d[3]
            self.rad_label = r_d[4]
            self.sensor = "Radar"
            self.cam_label = None
            self.cam_box = None
            self.cam_rad_iou = None
            self.cam_id = None
            self.rad_id = None
        elif r_d and c_d:
            self.cls = r_d[0]
            self.centroid = r_d[1]
            self.rad_box = r_d[2]
            self.rad_box2d = r_d[3]
            self.rad_label = r_d[4]
            self.cam_label = c_d[1]
            self.cam_box = c_d[0]
            self.cam_id = c_d[2]
            self.sensor = 'Both'
            self.cam_rad_iou = None
            self.rad_id = None
        elif trk:
            self.cls = None
            self.centroid = trk[1]
            self.rad_box = None
            self.rad_box2d = None
            self.rad_label = None
            self.cam_label = None
            self.cam_box = None
            self.cam_id = None
            self.sensor = 'Radar_track'
            self.cam_rad_iou = None
            self.rad_id = trk[0]
        self.radar_track = None


class DetectedObject_old:
    def __init__(self, cls):
        self.cls = cls
        self.cam_rad_iou = None
        self.rad_label = None
        self.rad_box = None
        self.rad_box2d = None
        self.centroid = None
        self.rad_id = None


class TrackedObjectALL:
    def __init__(self, c_id=None, r_id=None, sensor=None, id=None):
        self.id = id
        if c_id:
            self.c_id = [c_id]
            self.r_id = []
        elif r_id:
            self.c_id = []
            self.r_id = [r_id]
        elif c_id and r_id:
            self.c_id = [c_id]
            self.r_id = [r_id]
        self.life = 10
        self.activate = sensor
        self.deleted = False
        self.intrack = True

    def delete(self):
        self.c_id = []
        self.r_id = []
        self.activate = False
        self.life = 0
        self.deleted = True


class RadarTrackedObject:
    def __init__(self):
        self.dets = []
        self.start = None

    def get_prediction(self, camera=True):
        if camera:
            cam_label = [i.cam_label for i in self.dets]
            cam_box = [i.cam_box for i in self.dets]
            cam_id = [i.cam_id for i in self.dets if i.cam_id != None]
        r = [i.rad_label for i in self.dets]
        radar_label = np.empty((0, 4))
        for i in r:
            radar_label = np.vstack((radar_label, i))
        c = [i.centroid for i in self.dets]
        centroid = np.empty((0, 4))
        for i in c:
            centroid = np.vstack((centroid, i))

        num_pts = [len(i.cls) for i in self.dets]
        if camera:
            return radar_label, centroid, num_pts, cam_label, cam_box, cam_id
        else:
            return radar_label, centroid, num_pts


def timestamp_sync(imgs, t):
    diff = 10000
    ind = 0
    for i, img in enumerate(imgs):
        dt = abs(img - t)
        if dt < diff:
            diff = dt
            ind = i
    return ind, diff


def imgmsg_to_cv2(img_msg):
    dtype = np.dtype("uint8")  # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),
                              # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                              dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv


def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height  # That double line is actually integer division, not a comment
    return img_msg


def draw_gt_boxes3d(gt_boxes3d, fig, color=(1, 1, 1)):
    """
    Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (3,8) for XYZs of the box corners
        fig: figure handler
        color: RGB value tuple in range (0,1), box line color
    """
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],
                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, figure=fig)

        i, j = k + 4, (k + 1) % 4 + 4
        mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],
                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, figure=fig)

        i, j = k, k + 4
        mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],
                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, figure=fig)
    return fig


def cam_fov_pts(radar_pts, calib, img_width, img_height):
    # apply projection
    pts_2d = project_to_image(radar_pts.transpose(), calib)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (radar_pts[:, 0] > 0)
                    )[0]
    imgfov_pc_velo = radar_pts[inds, :]

    return imgfov_pc_velo


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick


def convert_polar(m):
    rg = np.square(m[:, :2])
    rg = np.sqrt(np.sum(rg, axis=1))
    theta = np.arctan2(m[:, 1], m[:, 0])
    measSet = np.vstack((rg, theta, m[:, 3]))
    return measSet


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.
    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    # assert isinstance(cloud, PointCloud2), 'cloud is not a sensor_msgs.msg.PointCloud2'
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step


def pc2_numpy(pc, l):
    arr = np.zeros((l, 5))
    pc2._type = 'sensor_msgs/PointCloud2'
    for i, point in enumerate(read_points(pc, skip_nans=True)):
        pt_x = point[0]
        pt_y = point[1]
        pt_z = point[2]
        doppler = point[3]
        # if pt_x!= 0 and pt_y != 0 and pt_z != 0:
        arr[i, 0] = pt_x
        arr[i, 1] = pt_y
        arr[i, 2] = pt_z
        arr[i, 3] = doppler
        # else:
        #     arr[i, 0] = 0.1
        #     arr[i, 1] = 0.1
        #     arr[i, 2] = -100
        #     arr[i, 3] = doppler
    return arr


def msg_sort(t):
    return t[0].header.stamp.sec + t[0].header.stamp.nanosec * 10 ** (-9)


def msg_sortv2(t):
    return t.header.stamp.sec + t.header.stamp.nanosec * 10 ** (-9)


