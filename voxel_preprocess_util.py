import numpy as np
import tensorflow as tf

labels = {
    "car": 0,
    "bus": 1,
    "person": 2,
    "truck": 3,
    "no_match": 4,
}


def quaternion_to_euler(w, x, y, z):
    """
    Converts quaternions with components w, x, y, z into a tuple (roll, pitch, yaw)

    """
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x ** 2 + y ** 2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y ** 2 + z ** 2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def to_transform_matrix(translation, rotation):
    Rt = np.eye(4)
    Rt[:3, :3] = eulerAnglesToRotationMatrix(rotation)
    Rt[:3, 3] = translation
    return Rt


def transform_lidar_box_3d(lidar, Rt):
    rt_inv = np.linalg.inv(Rt)

    lidar_3d = lidar[:, :3]
    lidar_3d = np.transpose(lidar_3d)

    ones = np.ones_like(lidar_3d[0])[None, :]
    hom_coord = np.concatenate((lidar_3d, ones), axis=0)
    lidar_3d = np.dot(rt_inv, hom_coord)
    lidar_3d = np.transpose(lidar_3d)[:, :3]

    return lidar_3d



def get_voxels_grid(voxel_size, grid_meters):
    voxel_size = np.asarray(voxel_size, np.float32)
    grid_size_meters = np.asarray(grid_meters, np.float32)
    voxels_grid = np.asarray(grid_size_meters / voxel_size, np.int32)
    return voxels_grid


def rot_z(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    ones = np.ones_like(c)
    zeros = np.zeros_like(c)
    return np.asarray([[c, -s, zeros], [s, c, zeros], [zeros, zeros, ones]])


def make_eight_points_boxes(bboxes_xyzlwhy):
    bboxes_xyzlwhy = np.asarray(bboxes_xyzlwhy)
    l = bboxes_xyzlwhy[:, 3] / 2.0
    w = bboxes_xyzlwhy[:, 4] / 2.0
    h = bboxes_xyzlwhy[:, 5] / 2.0
    # 3d bounding box corners
    x_corners = np.asarray([l, l, -l, -l, l, l, -l, -l])
    y_corners = np.asarray([w, -w, -w, w, w, -w, -w, w])
    z_corners = np.asarray([-h, -h, -h, -h, h, h, h, h])
    corners_3d = np.concatenate(([x_corners], [y_corners], [z_corners]), axis=0)
    yaw = np.asarray(bboxes_xyzlwhy[:, -1], dtype=np.float)
    corners_3d = np.transpose(corners_3d, (2, 0, 1))
    R = np.transpose(rot_z(yaw), (2, 0, 1))

    corners_3d = np.matmul(R, corners_3d)

    centroid = bboxes_xyzlwhy[:, :3]
    corners_3d += centroid[:, :, None]
    orient_p = (corners_3d[:, :, 0] + corners_3d[:, :, 7]) / 2.0
    orientation_3d = np.concatenate(
        (centroid[:, :, None], orient_p[:, :, None]), axis=-1
    )
    corners_3d = np.transpose(corners_3d, (0, 2, 1))
    orientation_3d = np.transpose(orientation_3d, (0, 2, 1))

    return corners_3d, orientation_3d


def get_bboxes_parameters_from_points(lidar_corners_3d):
    """
    The function returns 7 parameters of box [x, y, z, w, l, h, yaw]

    Arguments:
        lidar_corners_3d: [num_ponts, 8, 3]
    """
    centroid = (lidar_corners_3d[:, -2, :] + lidar_corners_3d[:, 0, :]) / 2.0
    delta_l = lidar_corners_3d[:, 0, :2] - lidar_corners_3d[:, 1, :2]
    delta_w = lidar_corners_3d[:, 1, :2] - lidar_corners_3d[:, 2, :2]
    width = np.linalg.norm(delta_w, axis=-1)
    length = np.linalg.norm(delta_l, axis=-1)

    height = lidar_corners_3d[:, -1, -1] - lidar_corners_3d[:, 0, -1]
    yaw = np.arctan2(delta_l[:, 1], delta_l[:, 0])

    return centroid, width, length, height, yaw


def get_bboxes_grid(bbox_labels, lidar_corners_3d, grid_meters, bbox_voxel_size):
    """
        The function transform lidar_corners_3d (8 points of bboxes) to
        parametrized version of bbox.
    """
    voxels_grid = get_voxels_grid(bbox_voxel_size, grid_meters)
    # Find box parameters
    centroid, width, length, height, _ = get_bboxes_parameters_from_points(
        lidar_corners_3d
    )
    # find the vector of orientation [centroid, orient_point]
    orient_point = (lidar_corners_3d[:, 1] + lidar_corners_3d[:, 2]) / 2.0

    voxel_coordinates = np.asarray(
        np.floor(centroid[:, :2] / bbox_voxel_size[:2]), np.int32
    )
    # Filter bboxes not fall in the grid
    bound_x = (voxel_coordinates[:, 0] >= 0) & (
        voxel_coordinates[:, 0] < voxels_grid[0]
    )
    bound_y = (voxel_coordinates[:, 1] >= 0) & (
        voxel_coordinates[:, 1] < voxels_grid[1]
    )
    mask = bound_x & bound_y
    # Filter all non related bboxes
    centroid = centroid[mask]
    orient_point = orient_point[mask]
    width = width[mask]
    length = length[mask]
    height = height[mask]
    bbox_labels = bbox_labels[mask]
    voxel_coordinates = voxel_coordinates[mask]
    # Confidence
    confidence = np.ones_like(width)

    # Voxels close corners to the coordinate system origin (0,0,0)
    voxels_close_corners = (
        np.asarray(voxel_coordinates, np.float32) * bbox_voxel_size[:2]
    )
    # Get x,y, coordinate
    delta_xy = centroid[:, :2] - voxels_close_corners
    orient_xy = orient_point[:, :2] - voxels_close_corners
    z_coord = centroid[:, -1]

    # print(
    #     f"confidence {confidence.shape}, delta_xy {delta_xy.shape}, orient_xy {orient_xy.shape}, z_coord {z_coord.shape}, width {width.shape}, height {height.shape}, bbox_labels {bbox_labels.shape}"
    # )
    # (x_grid, y_grid, (objectness, min_delta_x, min_delta_y, max_delta_x, max_delta_y, z, label))
    # objectness means 1 if box exists for this grid cell else 0
    output_tensor = np.zeros((voxels_grid[0], voxels_grid[1], 9), np.float32)
    if len(bbox_labels) > 0:
        data = np.concatenate(
            (
                confidence[:, None],
                delta_xy,
                orient_xy,
                z_coord[:, None],
                width[:, None],
                height[:, None],
                bbox_labels[:, None],
            ),
            axis=-1,
        )
        output_tensor[voxel_coordinates[:, 0], voxel_coordinates[:, 1]] = data
    return output_tensor


def get_boxes_from_box_grid(box_grid, bbox_voxel_size, conf_trhld=0.0):

    # Get non-zero voxels
    objectness, delta_xy, orient_xy, z_coord, width, height, label = tf.split(
        box_grid, (1, 2, 2, 1, 1, 1, -1), axis=-1
    )

    mask = box_grid[:, :, 0] > conf_trhld
    valid_idx = tf.where(mask)

    z_coord = tf.gather_nd(z_coord, valid_idx)
    width = tf.gather_nd(width, valid_idx)
    height = tf.gather_nd(height, valid_idx)
    objectness = tf.gather_nd(objectness, valid_idx)
    label = tf.gather_nd(label, valid_idx)
    delta_xy = tf.gather_nd(delta_xy, valid_idx)
    orient_xy = tf.gather_nd(orient_xy, valid_idx)
    voxels_close_corners = tf.cast(valid_idx, tf.float32) * bbox_voxel_size[None, :2]
    xy_coord = delta_xy + voxels_close_corners
    xy_orient = orient_xy + voxels_close_corners

    delta = xy_orient[:, :2] - xy_coord[:, :2]
    length = 2 * tf.norm(delta, axis=-1, keepdims=True)
    yaw = tf.expand_dims(tf.atan2(delta[:, 1], delta[:, 0]), axis=-1)

    bbox = tf.concat([xy_coord, z_coord, length, width, height, yaw], axis=-1,)
    return bbox, label, objectness


def make_top_view_image(lidar, grid_meters, voxels_size, channels=3):
    """
    The function makes top view image from lidar
    Arguments:
        lidar: lidar array of the shape [num_points, 3]
        width: width of the top view image
        height: height of the top view image
        channels: number of channels of the top view image
    """
    mask_x = (lidar[:, 0] >= 0) & (lidar[:, 0] < grid_meters[0])
    mask_y = (lidar[:, 1] >= 0) & (lidar[:, 1] < grid_meters[1])
    mask_z = (lidar[:, 2] >= 0) & (lidar[:, 2] < grid_meters[2])
    mask = mask_x & mask_y & mask_z
    lidar = lidar[mask]
    voxel_grid = get_voxels_grid(voxels_size, grid_meters)
    voxels = np.asarray(np.floor(lidar[:, :3] / voxels_size), np.int32)
    top_view = np.zeros((voxel_grid[0], voxel_grid[1], 2), np.float32)
    top_view[voxels[:, 0], voxels[:, 1], 0] = lidar[:, 2]  # z values
    top_view[voxels[:, 0], voxels[:, 1], 1] = lidar[:, 3]  # intensity values

    return top_view


def filter_boxes(labels, bboxes_3d, orient_3d, lidar, threshold=20):
    labels_res = []
    box_res = []
    orient_res = []
    for idx, box in enumerate(bboxes_3d):

        min_x = np.min(box[:, 0])
        max_x = np.max(box[:, 0])
        min_y = np.min(box[:, 1])
        max_y = np.max(box[:, 1])
        min_z = np.min(box[:, 2])
        max_z = np.max(box[:, 2])
        mask_x = (lidar[:, 0] >= min_x) & (lidar[:, 0] <= max_x)
        mask_y = (lidar[:, 1] >= min_y) & (lidar[:, 1] <= max_y)
        mask_z = (lidar[:, 2] >= min_z) & (lidar[:, 2] <= max_z)
        mask = mask_x & mask_y & mask_z
        result = np.sum(mask.astype(float))
        if result > threshold:
            box_res.append(box)
            orient_res.append(orient_3d[idx])
            labels_res.append(labels[idx])
    return np.asarray(labels_res), np.asarray(box_res), np.asarray(orient_res)


def save_lidar(lidar_filename, scan):
    scan = scan.reshape((-1))
    scan.tofile(lidar_filename)


def save_bboxes_to_file(
    filename, centroid, width, length, height, alpha, label, delim=";"
):

    if centroid is not None:
        with open(filename, "w") as the_file:
            for c, w, l, h, a, lbl in zip(
                centroid, width, length, height, alpha, label
            ):
                data = (
                    delim.join(
                        (
                            str(c[0]),
                            str(c[1]),
                            str(c[2]),
                            str(l),
                            str(w),
                            str(h),
                            str(a),
                            str(lbl),
                        )
                    )
                    + "\n"
                )
                # data = "{};{};{};{};{};{};{};{}\n".format(
                #     c[0], c[1], c[2], l, w, h, a, lbl
                # )
                the_file.write(data)


def get_color(label):
    # "car": 0, "Car": 1, "Misc": 2, "Van": 3, "Person_sitting": 4, "Pedestrian": 5, "Truck": 6, "Cyclist": 7
    color = np.asarray(
        [
            [255, 229, 204],  # "car": 0,
            [255, 255, 204],  # "bus": 1,
            [204, 204, 255],  # "person": 2,
            [255, 204, 204],  # "truck": 3,
            [255, 122, 204],  # "no_match": 4,
        ]
    )
    return color[int(label)]


def eulerAnglesToRotationMatrix(theta):

    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R