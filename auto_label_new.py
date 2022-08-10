import sys
sys.path.append('SVSF_Track')
run_cam_d = True
if run_cam_d:
    # sys.path.append('yolor')
    # from yolor.detect_custom import init_yoloR, detect
    # model, device, colors, names = init_yoloR(weights='yolor/yolor_p6.pt', cfg='yolor/cfg/yolor_p6.cfg',
    #                                            names='yolor/data/coco.names', out='inference/output', imgsz=1280, half=half)
    from Yolov5_StrongSORT_OSNet.track_custom import load_weight_sort, process_track
    sys.path.append('Yolov5_StrongSORT_OSNet')
    sys.path.append('Yolov5_StrongSORT_OSNet/strong_sort/deep/reid')
    device = '0'
    outputs = [None]
    device, model, stride, names, pt, imgsz, cfg, strongsort_list, \
    dt, seen, curr_frames, prev_frames, half = load_weight_sort(device,
                                                                'Yolov5_StrongSORT_OSNet/strong_sort/configs/strong_sort.yaml')
from radar_utils import *
import rosbag
from auto_label_util import *
import open3d as o3d

mtx = np.array([[770, 0., 639.5],
                [0.,770,381.6629],
                [0., 0., 1.]])

calib_pts = []
cam1 = np.empty((0,0))
v = (-108.20802358222203, 7.280529894768495, 470.76425650815855, ([12.091, -1.047, -2.0325]))




bag = rosbag.Bag("record/rooftop.bag")

# print(data)
all_bb = []
idx = 0
missidx = 0
# for car.bag
# rx = 1.56
# ry = 0
# rz = -.02
# tx = 0
# ty = 0
# tz = 0.1
# r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)

# for rooftop
rx = 1.56
ry = 0
rz = 0.025
tx = 0.2999999999999998
ty = 0
tz = 0
r2c_e = extrinsic_matrix(rx, ry, rz, tx, ty, tz)
c2g = extrinsic_matrix(-20/180*np.pi, 0, 0, 0, -10.9, 0)
g2c_p = cam_radar(20/180*np.pi, 0, 0, 0, 10.9, 0, mtx)
frame = SRS_data_frame()
epoch = None
root = 'dataset/rooftop'
p_arr_all = np.empty((0, 5))
pcd = o3d.geometry.PointCloud()
hull = np.array([[ 35.05376344,  25.97402597],
       [ 18.27956989,  -3.03030303],
       [ 87.52688172, -34.63203463],
       [ 93.11827957,  -9.95670996],
       [ 60.        ,   4.32900433],
       [ 63.01075269,  18.61471861]])
# hull_draw = np.zeros((hull.shape[0], hull.shape[1] + 1))
# hull_draw[:, :hull.shape[1]] = hull
# hull_draw = project_to_image(hull_draw.T, g2c_p)
# hull_image = []
# for i in hull_draw.shape[1]:
#     hull_image.append((hull_draw[]))
path = mpltPath.Path(hull)
for j, i in enumerate(bag.read_messages()):
    # read ros Topic camera or radar
    sensor = frame.load_data(i)
    if frame.full_data:
        file = open(f'dataset/rooftop/label/{idx:05d}.txt', 'w')
        # print(frame.camera.message.header.stamp.to_sec()- epoch)
        image_np = imgmsg_to_cv2(frame.camera.message)
        cv2.imwrite(f'dataset/rooftop/image/{idx:05d}.jpg', image_np)
        _, _, image_np_detection, camera_detection, outputs = process_track(image_np, i, curr_frames, prev_frames, outputs,
                                                             device, model, stride, names, pt, imgsz, cfg,
                                                             strongsort_list, dt,
                                                             seen, half, classes=[5, 7, 0, 2], conf_thres =0.10)
        npts = frame.radar.message.width
        arr_all = pc2_numpy(frame.radar.message, npts)
        arr_concat = np.vstack((arr_all, p_arr_all))
        arr_c = transform_radar(arr_concat.T, r2c_e).T  # from radar to camera (not pixel)
        arr_g = transform_radar(arr_c.T, c2g).T  # from camera to global
        xyz = arr_g[:, :3]
        pcd.points = o3d.utility.Vector3dVector(xyz)
        o3d.io.write_point_cloud(f"dataset/rooftop/radar/{idx:05d}.ply", pcd)
        # filter based on road mask
        mask = path.contains_points(np.vstack((arr_g[:, 2], arr_g[:, 0])).T)
        arr = arr_g[mask]
        # arr = filter_zero(arr_g)
        pc = arr[:, :4]
        total_box, cls = dbscan_cluster(pc, eps=2.5, min_sample=15)
        total_box_1, cls_1 = dbscan_cluster(pc, eps=2, min_sample=2)
        if isinstance(cls, type(None)):
            cls = []
        if isinstance(cls_1, type(None)):
            cls_1 = []
        total_box = np.vstack((total_box, total_box_1))
        box_index = non_max_suppression_fast(total_box[:, :4], .2)
        cls.extend(cls_1)
        cls = [cls[ii] for ii in box_index]
        total_box = total_box[box_index, :]
        image_np_detection, cam_arr = render_radar_on_image(arr_g, image_np_detection, g2c_p, 9000, 9000)
        if cls:
            for cc in cls:
                bbox = get_bbox_cls(cc)

                radar_detection = []
                for ii, cc in enumerate(cls):
                    centroid = np.mean(cc, axis=0)
                    # get tracking measurement
                    bbox = get_bbox_cls(cc)
                    bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)
                    bbox = project_to_image(bbox, g2c_p)
                    pts = project_to_image(cc.T, g2c_p)
                    box2d = get_bbox_2d(pts.T)
                    cv2.rectangle(image_np_detection, box2d[0], box2d[1], (255, 255, 0))
                    # cv2.rectangle(im0, box2d[0], box2d[1], (255, 255, 0))
                    box2d = [box2d[0][0], box2d[0][1], box2d[1][0], box2d[1][1]]
                    box2d = convert_topy_bottomx(box2d)

                    radar_detection.append([cc, centroid, bbox, box2d])
                radar_2d = np.asarray([ii[3] for ii in radar_detection])
                cam_2d = np.asarray([ii[0] for ii in camera_detection])
                if cam_2d.any():
                    radar_matched, camera_matched, ious, radar_unmatched, camera_unmatched = match_detection(radar_2d, cam_2d)
                    print(radar_matched)
                    for ii in range(len(radar_matched)):
                        c_matched = camera_detection[camera_matched[ii]]
                        r_matched = radar_detection[radar_matched[ii]]
                        label = get_bbox_cls_label(r_matched[0], c_matched[1])
                        file.write(' '.join([str(num) for num in label]))
                        file.write('\n')
                        cv2.putText(image_np_detection, f'Camera label: {c_matched[1]}', (
                            r_matched[-1][0], r_matched[-1][1]),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    for ii in range(len(radar_unmatched)):
                        r_matched = radar_detection[radar_unmatched[ii]]
                        label = get_bbox_cls_label(r_matched[0], 'no_match')
                        file.write(' '.join([str(num) for num in label]))
                        file.write('\n')
                        cv2.putText(image_np_detection, f'Camera label: no_match', (
                            r_matched[-1][0], r_matched[-1][1]),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 0, 255), 1, cv2.LINE_AA)

                # print(bbox)
                # bbox = get_bbox_coord(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], 0)
                #
                # bbox = project_to_image(bbox, g2c_p)
                # pts = project_to_image(cc.T, g2c_p)
                #
                # # print(pts)
                # box2d = get_bbox_2d(pts.T)
                # draw_projected_box3d(image_np_detection, bbox)
                # cv2.rectangle(image_np_detection, box2d[0], box2d[1], (255, 255, 0))
                # box2d = [box2d[0][0], box2d[0][1], box2d[1][0], box2d[1][1]]
                # box2d = convert_topy_bottomx(box2d)
                # matched = find_gt(box2d, detection)
                # label = get_bbox_cls_label(cc, matched[0][1])
                # if matched[0][1] == 'no_match':

                # print(hull_image)
                # cv2.drawContours(image_np_detection, [hull_image], 0, (255, 255, 255), 2)
                # bbframe.append(bbox)
                # file.write(' '.join([str(num) for num in label]))
                # file.write('\n')
                # if matched[0]:
                #     mbox2s = convert_xyxy(matched[0][0])
                #     cv2.rectangle(image_np, mbox2s[0], mbox2s[1], (255, 255, 0))
                #     cv2.imshow('no detection', image_np)
                #     cv2.waitKey(100)
        cv2.imshow('detection', image_np_detection)
        # cv2.imshow('no detection', image_np)
        cv2.waitKey(1)
        # idx+=1
        file.close()
        p_arr_all = arr_all.copy()
        idx+=1


print(idx)
print(missidx)