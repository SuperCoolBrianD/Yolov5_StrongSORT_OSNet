import os

import numpy as np

from radar_utils import *
import sys
sys.path.append('yolor')
sys.path.append('SVSF_Track')
from yolor.detect_custom import init_yoloR, detect
import rosbag
from auto_label_util import *
import pickle


# Read recording



half = False
radar_d = '/radar_data'
s = 0
model, device, colors, names = init_yoloR(weights='yolor/yolor_p6.pt', cfg='yolor/cfg/yolor_p6.cfg',
                                          names='yolor/data/coco.names', out='inference/output', imgsz=1280, half=False)

# adjust image visualization
cv2.namedWindow("Camera")
cv2.moveWindow('Camera', 800, 800)
# init loop


calib_pts = []
cam1 = np.empty((0,0))
v = (-108.20802358222203, 7.280529894768495, 470.76425650815855, ([12.091, -1.047, -2.0325]))




bag = rosbag.Bag("record/car.bag")

# print(data)
all_bb = []
idx = 0
missidx = 0

rx = 1.56
ry = 0
rz = -.02
tx = 0
ty = 0
tz = 0.1

mtx = np.array([[770, 0., 639.5],
                [0.,770,381.6629],
                [0., 0., 1.]])
def calib2str(mtx):
    p_2 = np.eye(4)
    p_2[:mtx.shape[0], :mtx.shape[1]] = mtx
    p_2 = p_2[:3, :]
    p_2 = p_2.flatten()
    p_2 = [str(i) for i in p_2]
    p_2 = ' '.join(p_2)
    return p_2
p_2 = calib2str(mtx)
v2c = extrinsic_matrix(rx, ry, rz, tx, ty, tz)
v2c = calib2str(v2c)
fill33 = np.eye(3)
fill43 = np.eye(4)
fill43 = calib2str(fill43)
fill33 = calib2str(fill33)
fillR = np.eye(3)
fillR = fillR.flatten()
fillR = [str(i) for i in fillR]
fillR = ' '.join(fillR)
p0 = ' '.join(['P0:', fill33])
p1 = ' '.join(['P1:', fill33])
p2 = ' '.join(['P2:', p_2])
p3 = ' '.join(['P3:', fill33])
r = ' '.join(['R0_rect:', fillR])
Tr_velo = ' '.join(['Tr_velo_to_cam:', fill43])
Tr_imu = ' '.join(['Tr_imu_to_velo:', fill43])
calib_file = '\n'.join([p0, p1, p2, p3, r, Tr_velo, Tr_imu])
print(calib_file)
r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)

frame = SRS_data_frame()
epoch = None
hull = np.array([[-18.06451613,  13.63636364],
       [ 48.06451613,  81.6017316 ],
       [ 59.03225806,  69.04761905],
       [ -7.74193548,  -1.51515152]])
for j, i in enumerate(bag.read_messages()):
    # read ros Topic camera or radar
    sensor = frame.load_data(i)
    # print(idx)
    # print(sensor)
    # if sensor == '/Radar':
    #     os.makedirs(f'kitti_dataset/{idx:06d}')
    #     file = open(f'kitti_dataset/{idx:06d}/ground_truth.txt', 'w')
    #     idx+=1
    if frame.full_data:
        figs, axs = plt.subplots(1, figsize=(6, 6))
        file = open(f'kitti_dataset/label_2/{idx:06d}.txt', 'w')
        c_file = open(f'kitti_dataset/calib/{idx:06d}.txt', 'w')
        c_file.write(calib_file)
        c_file.close()
        print(abs(abs(frame.camera.message.header.stamp.to_sec()- frame.radar.message.header.stamp.to_sec())-1))
        # print(frame.camera.message.header.stamp.to_sec()- epoch)
        image_np = imgmsg_to_cv2(frame.camera.message)
        cv2.imwrite(f'kitti_dataset/image_2/{idx:06d}.png', image_np)
        npts = frame.radar.message.width
        arr_all = pc2_numpy(frame.radar.message, npts)
        radar_pc = np2bin(arr_all)
        radar_pc.astype('float32').tofile(f'kitti_dataset/velodyne/{idx:06}.bin')
        path = mpltPath.Path(hull)
        mask = path.contains_points(arr_all[:, :2])
        arr_all = arr_all[mask]
        # # draw points on plt figure
        # arr = filter_zero(arr_all)
        pc = arr_all[:, :4]
        total_box, cls = dbscan_cluster(pc, eps=2.5, min_sample=15)
        total_box_1, cls_1 = dbscan_cluster(pc, eps=2, min_sample=6)
        total_box = np.vstack((total_box, total_box_1))
        box_index = non_max_suppression_fast(total_box[:, :4], .2)
        cls.extend(cls_1)
        cls = [cls[ii] for ii in box_index]
        total_box = total_box[box_index, :]
        image_np_detection, detection = detect(source=image_np, model=model, device=device, colors=colors, names=names,
                                 view_img=False, half=half)
        image_np_detection, cam_arr = render_radar_on_image(arr_all, image_np_detection, r2c, 9000, 9000)
        if cls:
            for cc in cls:
                bbox = get_bbox_cls(cc)
                # print(bbox)
                bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)

                bbox = project_to_image(bbox, r2c)
                pts = project_to_image(cc.T, r2c)
                # print(pts)
                box2d = get_bbox_2d(pts.T)
                draw_projected_box3d(image_np_detection, bbox)
                cv2.rectangle(image_np_detection, box2d[0], box2d[1], (255, 255, 0))
                cv2.rectangle(image_np_detection, box2d[0], box2d[1], (255, 255, 0))
                box2d = [box2d[0][0], box2d[0][1], box2d[1][0], box2d[1][1]]
                box2d = convert_topy_bottomx(box2d)
                matched = find_gt(box2d, detection)
                label = get_bbox_cls_label_kitti(cc, matched[0][1], box2d)
                # bbframe.append(bbox)
                file.write(' '.join([str(num) for num in label]))
                file.write('\n')
                # if matched[0]:
                #     mbox2s = convert_xyxy(matched[0][0])
                #     cv2.rectangle(image_np, mbox2s[0], mbox2s[1], (255, 255, 0))
                #     cv2.imshow('no detection', image_np)
                #     cv2.waitKey(100)
        # cv2.imshow('detection', image_np_detection)
        # cv2.imshow('no detection', image_np)
        # cv2.waitKey(1)
        # idx+=1
        # frame.clear_data()
        file.close()
        idx += 1
        print(idx)


print(idx)
print(missidx)