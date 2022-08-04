import matplotlib
matplotlib.use('TkAgg')
# import mayavi.mlab as mlab

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image
import sys
import math
import matplotlib.path as mpltPath


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
    rr = rx@ry@rz
    r = np.eye(4)
    r[:3, :3] = rr
    return r@t


def c_extrinsic_matrix(rx, ry, rz, tx, ty, tz):
    rx = rotx(rx)
    ry = roty(ry)
    rz = rotz(rz)
    R = rx@ry@rz
    C = np.array([[tx, ty, tz]]).T
    t = -R@C
    mtr = np.eye(4)
    mtr[:3, :3] = R.T
    mtr[:3, 2] = t
    return mtr



def cam_radar(rx, ry, rz, tx, ty, tz, c):
    cam_matrix = np.eye(4)
    cam_matrix[:3, :3] = c
    extrinsic = extrinsic_matrix(rx, ry, rz, tx, ty, tz)

    proj_radar2cam = cam_matrix@extrinsic

    return proj_radar2cam


def camera_pose(rx, ry, rz, tx, ty, tz, c):
    cam_matrix = np.eye(4)
    cam_matrix[:3, :3] = c
    extrinsic = extrinsic_matrix(rx, ry, rz, tx, ty, tz)
    # print(radar_matrix)
    proj_radar2cam = cam_matrix@extrinsic
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
        if cl>255:
            cl=255
        elif cl<0:
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
        return np.zeros([1,5])
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


def filter_zero(pc):
    """Filter low velocity points"""
    mask = np.abs(pc[:, 3]) > 0.05
    pc = pc[mask, :]
    return pc


def get_bbox(arr):
    x_coord, y_coord, z_coord = arr[:, 0], arr[:, 1], arr[:, 2]
    x_min = min(x_coord)-1
    y_min = min(y_coord)-1
    x_max = max(x_coord)+1
    y_max = max(y_coord)+1
    return [x_min, y_min, x_max-x_min, y_max-y_min], np.array([[x_min, y_min, x_max, y_max, 1]])


def get_bbox_2d(arr):
    x_coord, y_coord, z_coord = arr[:, 0], arr[:, 1], arr[:, 2]
    x_min = min(x_coord)-1
    y_min = min(y_coord)-1
    x_max = max(x_coord)+1
    y_max = max(y_coord)+1
    return (int(x_min), int(y_max)), (int(x_max), int(y_min))


def get_bbox_cls(arr):
    x_coord, y_coord, z_coord = arr[:, 0], arr[:, 1], arr[:, 2]
    x_min = min(x_coord)
    x_max = max(x_coord)
    y_min = min(y_coord)
    y_max = max(y_coord)
    z_min = min(z_coord)
    z_max = max(z_coord)
    return np.array([x_min+(x_max-x_min)/2, y_min+(y_max-y_min)/2, z_min+(z_max-z_min)/2,
                      x_max-x_min, y_max-y_min, z_max-z_min, 0])


def get_bbox_cls_label(arr, clf):
    x_coord, y_coord, z_coord = arr[:, 0], arr[:, 1], arr[:, 2]
    x_min = min(x_coord)
    x_max = max(x_coord)
    y_min = min(y_coord)
    y_max = max(y_coord)
    z_min = min(z_coord)
    z_max = max(z_coord)
    return [x_min+(x_max-x_min)/2, y_min+(y_max-y_min)/2, z_min+(z_max-z_min)/2,
                      x_max-x_min, y_max-y_min, z_max-z_min, 0, 0, 0, clf]



def get_bbox_cls_label_kitti(arr, clf, box2d):
    x_coord, y_coord, z_coord = arr[:, 0], arr[:, 1], arr[:, 2]
    x_min = min(x_coord)
    x_max = max(x_coord)
    y_min = min(y_coord)
    y_max = max(y_coord)
    z_min = min(z_coord)
    z_max = max(z_coord)
    return [clf,0, 0, 0, box2d[0], box2d[1], box2d[2], box2d[3],
                      x_max-x_min, y_max-y_min, z_max-z_min, x_min+(x_max-x_min)/2, y_min+(y_max-y_min)/2, z_min+(z_max-z_min)/2, 0, 0]



def get_bbox_coord(t1, t2, t3, w, h, l, rz, is_homogenous=False):
    # 3d bounding box dimensions

    # 3D bounding box vertices [3, 8]
    z = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
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
    axs: plotting parameter set to None if no required
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
        # if axs:
        #     axs.scatter(c[:, 0], c[:, 1], s=0.5)
        #     rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='blue',
        #                              facecolor='none')
        #     axs.add_patch(rect)
            # axs.text(c[0, 0], c[0, 1], f"cluster {i}", fontsize=11,
            # color='r')
        #     # axs.text(c[0, 0], c[0, 1]-5, f"{c.shape[0]} points", fontsize=11,
        #     # color='b')
    return total_box, cls


def plot_box(bbox, c, axs):
    # axs.scatter(c[:, 0], c[:, 1], s=0.5)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]- bbox[1], linewidth=1, edgecolor='blue',
                             facecolor='none')
    axs.add_patch(rect)


def get_centroid(bbox):
    return bbox[2]-bbox[0], bbox[3]-bbox[1]


def match_measurement(detection_list, tracks):
    distance = 999
    picked = None
    thresh = 3
    for index, det in enumerate(detection_list):
        if det.sensor != 'Camera':
            d = np.sqrt((det.centroid[0] - tracks[0])**2 + (det.centroid[1] - tracks[1])**2)
            if d < distance and d < thresh:
                distance = d
                picked = index
    return picked


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
            for i in range(2, average_window+1):
                speed_hist.append(np.sqrt(((self.tracklet[-1][0][0] - self.tracklet[-i][0][0]) ** 2 +
                                      (self.tracklet[-1][0][1] - self.tracklet[-i][0][1]) ** 2)) / (
                                         self.tracklet[-1][1] - self.tracklet[-i][1]) * 3.6)
            self.speed = sum(speed_hist) / len(speed_hist)
        else:
            self.speed = 'Measuring'


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
            self.radar_frame +=1
        elif data.topic == '/Camera':
            self.camera = data
            self.has_camera = True
            self.camera_frame+=1
        if self.has_radar and self.has_camera:
            self.full_data = True
        return data.topic


    def load_data_radar_only(self, data):
        # if self.full_data:
        #     self.clear_data()
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

    def clear_data(self):
        self.camera = None
        self.radar = None
        self.has_radar = False
        self.has_camera = False
        self.full_data = False


class DetectedObject:
    def __init__(self, r_d=None, c_d=None):
        if c_d and not  r_d:
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


class DetectedObject_old:
    def __init__(self, cls):
        self.cls = cls
        self.cam_rad_iou = None
        self.rad_label = None
        self.rad_box = None
        self.rad_box2d = None
        self.centroid = None
        self.rad_id = None

class TrackedObject:
    def __init__(self):
        self.dets=[]
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
            return radar_label, centroid,  num_pts

def timestamp_sync(imgs, t):
    diff = 10000
    ind = 0
    for i, img in enumerate(imgs):
        dt = abs(img - t)
        if dt < diff:
            diff = dt
            ind = i
    return ind, diff


def pc2_numpy(pc, l):
    arr = np.zeros((l, 5))
    for i ,point in enumerate(pc2.read_points(pc, skip_nans=True)):
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


def imgmsg_to_cv2(img_msg):
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
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
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
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
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds


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
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
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