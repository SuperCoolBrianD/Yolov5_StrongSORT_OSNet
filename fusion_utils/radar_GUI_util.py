import matplotlib
matplotlib.use('TkAgg')
# import mayavi.mlab as mlab

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
from sensor_msgs.msg import Image
import sys
sys.path.append('yolor')
sys.path.append('SVSF_Track')
import math
import matplotlib.path as mpltPath
from fusion_utils.radar_utils import *
import yaml
from yaml import Loader
from mpl_point_clicker import clicker
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument('--cfg_file', default='cfg.yaml')
    parser.add_argument('--footage', default='record/rooftop.bag')
    return parser

def empty():
    pass


def get_img_local(event, x, y, flags, param):
    global nxt
    global img_coord
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        nxt = True
        img_coord = (x, y)

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

def cfg_write(filename, r2c_ext, c2g_ext, intrinsic, road_mask_x, road_mask_y):
    input_file = {'radar_to_cam_extrinsic': {'matrix': r2c_ext}, 'cam_to_WCS_extrinsic': {'matrix': c2g_ext},
                  'cam_intrinsic': {'matrix': intrinsic},
                  'road_mask': {'array': {'x': road_mask_x, 'y': road_mask_y}}}
    with open(filename, 'w') as f:
        data = yaml.dump(input_file, f)
    return

def r2c_extrinsic(intrinsic, footage):
    # Read recording
    bag = rosbag.Bag(footage)
    topics = bag.get_type_and_topic_info()

    cv2.namedWindow("Camera")
    cv2.moveWindow('Camera', 800, 800)
    # intrinsic
    mtx = np.eye(3)
    mtx[0, :] = intrinsic[0:3]
    mtx[1, :] = intrinsic[3:6]
    mtx[2, :] = intrinsic[6:]

    cv2.setMouseCallback('Camera', get_img_local)
    cam1 = np.empty((0, 0))
    cv2.namedWindow('TrackBar')
    cv2.resizeWindow('TrackBar', 640, 320)
    cv2.createTrackbar('rx', 'TrackBar', 0, 314, empty)
    cv2.createTrackbar('ry', 'TrackBar', 0, 314, empty)
    cv2.createTrackbar('rz', 'TrackBar', 0, 314, empty)
    cv2.createTrackbar('tx', 'TrackBar', 0, 100, empty)
    cv2.createTrackbar('ty', 'TrackBar', 0, 100, empty)
    cv2.createTrackbar('tz', 'TrackBar', 0, 100, empty)
    cv2.setTrackbarPos('rx', 'TrackBar', 158, )
    cv2.setTrackbarPos('ry', 'TrackBar', 0, )
    cv2.setTrackbarPos('rz', 'TrackBar', 162, )
    cv2.setTrackbarPos('tx', 'TrackBar', 57, )
    cv2.setTrackbarPos('ty', 'TrackBar', 50, )
    cv2.setTrackbarPos('tz', 'TrackBar', 50, )
    frame = SRS_data_frame()
    r2c_ext = [0, 0, 0, 0, 0, 0]
    for j, i in enumerate(bag.read_messages()):
        sensor = frame.load_data(i)
        if sensor == "/Radar":
            npts = frame.radar.message.width
            arr_all = pc2_numpy(frame.radar.message, npts)
            # arr_concat = np.vstack((arr_all, p_arr_all))
            arr_concat = arr_all
            p_arr_all = arr_concat.copy()
        # print(idx)
        # print(sensor)
        if frame.full_data:
            image_np = imgmsg_to_cv2(frame.camera.message)
            arr = filter_zero(arr_all)
            # draw points on plt figure
            pc = arr[:, :4]
            ped_box = np.empty((0, 5))
            total_box, cls = dbscan_cluster(pc, eps=2, min_sample=20)
            if total_box.any() and ped_box.any:
                total_box = np.vstack((total_box, ped_box))
            # yolo detection
            # cam1, detection = detect(source=cam1, model=model, device=device, colors=colors, names=names,
            #                              view_img=False)
            # Radar projection onto camera parameters
            rx = cv2.getTrackbarPos('rx', 'TrackBar') / 100
            ry = cv2.getTrackbarPos('ry', 'TrackBar') / 100
            rz = cv2.getTrackbarPos('rz', 'TrackBar') / 100 - 157
            tx = cv2.getTrackbarPos('tx', 'TrackBar') / 10 - 5
            ty = cv2.getTrackbarPos('ty', 'TrackBar') / 10 - 5
            tz = cv2.getTrackbarPos('tz', 'TrackBar') / 10 - 5

            r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)
            new_cam1, cam_arr = render_radar_on_image(arr, cam1, r2c, 9000, 9000)


            print('Adjust using trackbar, Press c for next frame')
            print('Press q to return values')
            while True:
                rx = cv2.getTrackbarPos('rx', 'TrackBar') / 100
                ry = cv2.getTrackbarPos('ry', 'TrackBar') / 100
                rz = cv2.getTrackbarPos('rz', 'TrackBar') / 100 - 1.57
                tx = cv2.getTrackbarPos('tx', 'TrackBar') / 10 - 5
                ty = cv2.getTrackbarPos('ty', 'TrackBar') / 10 - 5
                tz = cv2.getTrackbarPos('tz', 'TrackBar') / 10 - 5
                r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)
                r2c_ext[0] = rx
                r2c_ext[1] = ry
                r2c_ext[2] = rz
                r2c_ext[3] = tx
                r2c_ext[4] = ty
                r2c_ext[5] = tz

                # extrinsic radar -> pixel coordinate
                # radar -> camera coordinate
                # radar_cam_coord -> rotx(alpha) * radar_cam_coord -> world coordinate with origin at radar (pitch about 5 degree)
                new_cam1, cam_arr = render_radar_on_image(arr_all, image_np, r2c, 9000, 9000)
                if cls:
                    for cc in cls:
                        bbox = get_bbox_cls(cc)
                        bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)
                        bbox = project_to_image(bbox, r2c)
                        draw_projected_box3d(new_cam1, bbox)
                        xyz = np.mean(cc, axis=0).reshape((-1, 1))
                        xyz = xyz[:3, :]
                        cent = project_to_image(xyz, r2c)
                        cent = (int(cent[0, 0]), int(cent[1, 0]))
                        new_cam1 = cv2.circle(new_cam1, cent, 5, (255, 255, 0), thickness=2)

                cv2.imshow('Camera', new_cam1)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    break
                if key == ord('q'):
                    cv2.destroyWindow('TrackBar')
                    cv2.destroyWindow('Camera')
                    return r2c_ext




def c2g_extrinsic(r2c_ext, footage):
    # Read recording
    bag = rosbag.Bag(footage)
    topics = bag.get_type_and_topic_info()

    rx = r2c_ext[0]
    ry = r2c_ext[1]
    rz = r2c_ext[2]
    tx = r2c_ext[3]
    ty = r2c_ext[4]
    tz = r2c_ext[5]

    r2c_e = extrinsic_matrix(rx, ry, rz, tx, ty, tz)
    frame = SRS_data_frame()
    mes = []
    fig, axs = plt.subplots(2)
    for j, i in enumerate(bag.read_messages()):
        if j == 1000:
            break
        sensor = frame.load_data(i)                                 #### cannot delete
        if frame.full_data:
            image_np = imgmsg_to_cv2(frame.camera.message)
            npts = frame.radar.message.width
            arr_all_rad = pc2_numpy(frame.radar.message, npts)
            arr_all = transform_radar(arr_all_rad.T, r2c_e).T
            moving = filter_zero(arr_all)
            pc = moving[:, :4]
            total_box_1, cls_1 = dbscan_cluster(pc, eps=2, min_sample=2)
            if cls_1:
                num = np.random.uniform()
                if num <= 0.5:
                    for ii in cls_1:
                        xyz = np.mean(ii, axis=0)
                        x = xyz[0]
                        z = xyz[2]
                        miny = ii[1, :][np.argmin(ii[1, :])]
                        rng = math.sqrt(z ** 2 + miny ** 2)
                        data = [x, miny, z, rng]
                        mes.append(data)

            cv2.imshow('camera', image_np)
            cv2.waitKey(1)

    cv2.destroyWindow('camera')
    h_list = []
    d_list = []
    for i, d1 in enumerate(mes):
        for d2 in mes[i + 1:]:
            tana = (d1[2] - d2[2]) / (d2[1] - d1[1])
            alpha = abs(math.atan(tana))
            theta = math.sin(d2[1] / d2[3])
            if alpha > np.pi / 4 and alpha - theta < np.pi / 2:
                h = d2[3] * math.cos(alpha - theta)
                d_list.append(alpha)
                h_list.append(h)

    angle_n, angle_bins, angle_patches = axs[0].hist(d_list, 200, facecolor='g', alpha=0.75)
    height_n, height_bins, height_patches = axs[1].hist(h_list, 200, facecolor='g', alpha=0.75)
    height_index = np.argmax(height_n)
    height = height_bins[height_index]
    angle_index = np.argmax(angle_n)
    angle = angle_bins[angle_index]
    axs[0].set_title('Mounting Angle Histogram')
    axs[0].set_xlabel("Angle (rad)")
    axs[1].set_title('Mounting Height Histogram')
    axs[1].set_xlabel("Height (m)")
    plt.show()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        plt.close()
    angle = np.pi/2-angle
    extrinsic = [-float(angle), 0, 0, 0, -float(height), 0]
    print(extrinsic)
    return extrinsic

def road_mask(intrinsic, r2c_ext, c2g_ext, footage):
    bag = rosbag.Bag(footage)
    # bag = rosbag.Bag("record/traffic1.bag")
    topics = bag.get_type_and_topic_info()

    frame = SRS_data_frame()
    p_arr_all = np.empty((0, 5))

    mtx = np.eye(3)
    mtx[0, :] = intrinsic[0:3]
    mtx[1, :] = intrinsic[3:6]
    mtx[2, :] = intrinsic[6:]

    rx = r2c_ext[0]
    ry = r2c_ext[1]
    rz = r2c_ext[2]
    tx = r2c_ext[3]
    ty = r2c_ext[4]
    tz = r2c_ext[5]

    r2c_e = extrinsic_matrix(rx, ry, rz, tx, ty, tz)
    c2g = extrinsic_matrix(c2g_ext[0], c2g_ext[1], c2g_ext[2], c2g_ext[3], c2g_ext[4], c2g_ext[5])
    r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)

    total_radar_move = np.empty((0, 5))
    total_radar_zero = np.empty((0, 5))

    for j, i in enumerate(bag.read_messages()):
        if j == 2000:
            break
        sensor = frame.load_data(i)
        if frame.full_data:
            # print(frame.radar.message.header.stamp.to_sec()- epoch)
            # epoch = frame.radar.message.header.stamp.to_sec()
            image_np = imgmsg_to_cv2(frame.camera.message)
            npts = frame.radar.message.width
            arr_all = pc2_numpy(frame.radar.message, npts)
            arr_concat = np.vstack((arr_all, p_arr_all))
            # coordinate transformation from radar to global
            arr_c = transform_radar(arr_concat.T, r2c_e).T  # from radar to camera (not pixel)
            arr_g = transform_radar(arr_c.T, c2g).T  # from camera to global
            arr_non_zero = filter_zero(arr_g)
            arr_zero = filter_move(arr_g)
            total_radar_move = np.vstack((total_radar_move, arr_non_zero))
            total_radar_zero = np.vstack((total_radar_zero, arr_zero))
            cv2.imshow('camera', image_np)
            cv2.waitKey(1)
            # input()
            p_arr_all = arr_all.copy()

    figs, axs = plt.subplots(1, figsize=(6, 6))
    klicker = clicker(axs, ["event"], markers=["x"], **{"linestyle": "--"})

    axs.set_xlim(-100, 100)
    axs.set_ylim(-100, 100)
    axs.scatter(total_radar_zero[:, 0], total_radar_zero[:, 2], s=0.5, color='r')
    axs.scatter(total_radar_move[:, 0], total_radar_move[:, 2], s=1)
    plt.show()
    hull = klicker.get_positions()
    lst = hull.get("event")
    road_mask_x = []
    road_mask_y = []
    for j in range(lst.shape[0]):
        road_mask_x.append(lst[j, 0])
        road_mask_y.append(lst[j, 1])
    road_mask_x = [float(i) for i in road_mask_x]
    road_mask_y = [float(i) for i in road_mask_y]

    return road_mask_x, road_mask_y
    #result found in the histogram
# mtx = np.array([[747.9932, 0., 655.5036],
#                 [0., 746.6126, 390.1168],
#                 [0., 0., 1.]])
#intrinsic = [747.9932, 0., 655.5036,0., 746.6126, 390.1168,0., 0., 1.]
#r2c_ext = [1.56, 0, 0.05, 0.3, 0, 0]
#c2g_ext = [-20/180*np.pi, 0, 0, 0, -10.9, 0]
# road_mask(intrinsic, r2c_ext, c2g_ext, "record/rooftop.bag")
#print(r2c_extrinsic(intrinsic, r2c_ext, "record/rooftop.bag"))