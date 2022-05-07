import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches

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


def radar_cam(rx, ry, rz, tx, ty, tz):
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


def cam_radar(rx, ry, rz, tx, ty, tz, c):
    cam_matrix = np.eye(4)
    cam_matrix[:3, :3] = c
    # cam_matrix[0, 0] = 640
    # cam_matrix[1, 1] = 480
    # cam_matrix[0, 2] = 320
    # cam_matrix[1, 2] = 240
    radar_matrix = radar_cam(rx, ry, rz, tx, ty, tz)
    proj_radar2cam = cam_matrix@radar_matrix
    return proj_radar2cam

def render_radar_on_image(pts_radar, img, proj_radar2cam, img_width, img_height):
    """functions to project radar points on image"""
    # projection matrix (project from velo2cam2)
    # todo change to use actual camera intrinsic from calibration
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


def get_bbox_cls(arr):
    x_coord, y_coord, z_coord = arr[:, 0], arr[:, 1], arr[:, 2]
    x_min = min(x_coord)
    y_min = min(y_coord)
    x_max = max(x_coord)
    y_max = max(y_coord)
    z_min = min(z_coord)
    z_max = max(z_coord)
    return np.array([x_min+(x_max-x_min)/2, y_min+(y_max-y_min)/2, z_min+(z_max-z_min)/2 ,
                      x_max-x_min, y_max-y_min, z_max-z_min, 0])


def in_camera_coordinate(t1, t2, t3, l, w, h, ry, is_homogenous=False):
    # 3d bounding box dimensions

    # 3D bounding box vertices [3, 8]
    x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    z = [0, 0, 0, 0, -h, -h, -h, -h]
    y = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    box_coord = np.vstack([x, y, z])
    R = roty(ry)  # [3, 3]
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
    clustering.fit(pc)
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
        if axs:
            axs.scatter(c[:, 0], c[:, 1], s=0.5)
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='blue',
                                     facecolor='none')
            axs.add_patch(rect)
            # axs.text(c[0, 0], c[0, 1], f"cluster {i}", fontsize=11,
            # color='r')
            # axs.text(c[0, 0], c[0, 1]-5, f"{c.shape[0]} points", fontsize=11,
            # color='b')
    return total_box, cls


def get_centroid(bbox):
    return bbox[2]-bbox[0], bbox[3]-bbox[1]


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


def timestamp_sync(imgs, t):
    diff = 10000
    ind = 0
    for i, img in enumerate(imgs):
        dt = abs(img - t)
        if dt < diff:
            diff = dt
            ind = i
    return ind, diff
