import numpy as np
import pickle
import os
from tqdm import tqdm
import argparse
from voxel_preprocess_util import (
    quaternion_to_euler,
    to_transform_matrix,
    transform_lidar_box_3d,
    make_eight_points_boxes,
    labels as cls_dict,
    filter_boxes,
    get_bboxes_parameters_from_points,
    save_lidar,
    save_bboxes_to_file
)

def read_label(file_name):
    file = open(file_name, 'r')
    label = np.empty((0, 8))

    for line in file:
        bbox = line.split(' ')
        cls = cls_dict[bbox[-1].strip('\n')]
        bbox = [float(i) for i in bbox[:7]]
        bbox.insert(0, cls)
        label = np.vstack((label, bbox))
    return label


def make_xzyhwly(bboxes):
    """
    Get raw data from bboxes and return xyzwlhy
    """
    label = bboxes[:, 0]
    c_x = bboxes[:, 1]
    c_y = bboxes[:, 2]
    c_z = bboxes[:, 3]
    length = bboxes[:, 4]
    width = bboxes[:, 5]
    height = bboxes[:, 6]
    yaw = bboxes[:, 7]
    new_boxes = np.asarray([c_x, c_y, c_z, length, width, height, yaw], dtype=np.float)
    return label, np.transpose(new_boxes)

def preprocess_data(dataset_dir):
    """
    The function prepares data for training from pandaset.
    Arguments:
        dataset_dir: directory with  Pandaset data
    """

    # Get list of data samples
    seq_list = os.listdir('../dataset')
    for seq in tqdm(range(len(seq_list)), desc="Process sequences", total=len(seq_list)):
        radar_path = f"dataset/{seq:05d}/radar.pkl"
        radar = pickle.load(open(radar_path,'rb'))
        # leave only doppler
        radar = radar[:, :4]
        bboxes = read_label(f"dataset/{seq:05d}/ground_truth.txt")
        if not bboxes.any():
            lidar_filename = os.path.join(f"dataset/{seq:05d}", 'radar_processed' + ".bin")
            save_lidar(lidar_filename, radar.astype(np.float32))
            box_filename = os.path.join(f"dataset/{seq:05d}", 'boxes_processed' + ".txt")
            open(box_filename, 'w')
            continue
        labels, bboxes = make_xzyhwly(bboxes)
        corners_3d, orientation_3d = make_eight_points_boxes(bboxes)
        # # filter boxes containing less then 20 lidar points inside
        labels, corners_3d, orientation_3d = filter_boxes(
            labels, corners_3d, orientation_3d, radar, threshold=5
        )
        centroid, width, length, height, yaw = get_bboxes_parameters_from_points(
            corners_3d
        )
        lidar_filename = os.path.join(f"dataset/{seq:05d}", 'radar_processed' + ".bin")
        save_lidar(lidar_filename, radar.astype(np.float32))
        box_filename = os.path.join(f"dataset/{seq:05d}", 'boxes_processed' + ".txt")
        save_bboxes_to_file(
            box_filename, centroid, width, length, height, yaw, labels
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess 3D traffic radar dataset.")
    parser.add_argument("--dataset_dir", default="dataset")
    args = parser.parse_args()
    preprocess_data(args.dataset_dir)

# centroid, width, length, height, yaw = get_bboxes_parameters_from_points(
#     corners_3d
# )
#
# # Save data
# lidar_filename = os.path.join(lidar_out_dir, sample_idx + ".bin")
# save_lidar(lidar_filename, radar.astype(np.float32))
# box_filename = os.path.join(bbox_out_dir, sample_idx + ".txt")
# save_bboxes_to_file(
#     box_filename, centroid, width, length, height, yaw, labels
# )
#
#
