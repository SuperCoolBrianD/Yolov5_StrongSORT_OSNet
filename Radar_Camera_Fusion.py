import sys

sys.path.append('SVSF_Track')
run_cam_d = True
if run_cam_d:
    from Yolov5_StrongSORT_OSNet.track_custom import load_weight_sort, process_track
    sys.path.append('Yolov5_StrongSORT_OSNet')
    sys.path.append('Yolov5_StrongSORT_OSNet/strong_sort/deep/reid')
    device = '0'
    outputs = [None]
    device, model, stride, names, pt, imgsz, cfg, strongsort_list, \
    dt, seen, curr_frames, prev_frames, half = load_weight_sort(device,
                                                                'Yolov5_StrongSORT_OSNet/strong_sort/configs/strong_sort.yaml')



import rosbag
from SVSF_Track.MTT_Functions import *
from SVSF_Track.track import track
import pickle
from fusion_utils.radar_utils import *
from matplotlib.animation import FuncAnimation
from fusion_utils.learn_utils import *
from fusion_utils.auto_label_util import *
import matplotlib
from fusion_utils.radar_GUI_util import cfg_read
matplotlib.use('TkAgg')
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import fusion_utils.detection_utils as detection_utils
radar_track = detection_utils.RadarTracking()

import time
# Read recording
fig, axs = plt.subplots(1, figsize=(6, 6))
fig.canvas.set_window_title('Radar Detection and Tracking IMM_small')
s = 0
cv2.imshow('Camera', np.zeros((480, 640)))
cv2.moveWindow('Camera', 800, 800)
image_np = np.empty((5, 5))
# init loop
radar_time_0 = 0
camera_time_0 = 0
update = 1
cam1 = np.empty((5,5))
tracked_object_radar = dict()
tracked_object_all = []
alive_track = [0] * 8000
life = 10
# track parameters
# init KF tracker


radar_p_time = 0
t_array = []
dt_array = []

cam_msg = 0

svm_model = pickle.load(open('radar_model_zoo/svm_model_scale.pkl', 'rb'))
scaler_model = pickle.load(open('radar_model_zoo/scaler_model.pkl', 'rb'))
classes = ['car', 'bus', 'person', 'truck', 'no_match']
yolo_classes = ['car', 'bus', 'person', 'truck', 'no_match']
(rx, ry, rz, tx, ty, tz), (alpha, _, _, _, height, _), mtx, hull  = cfg_read('config/remote.yaml')
parameter = calib(rx, ry, rz, tx, ty, tz, mtx, alpha, height)
r2c_e = parameter.r2c_e
c2g = parameter.c2g
g2c_p = parameter.g2c_p
c2wcs = parameter.c2wcs
idx=0
epoch = 0
cluster_hist = [[] for i in range(8000)]
class_track = [[] for i in range(8000)]
tracked_list = [RadarTrackedObject() for i in range(8000)]
bus_count = 0
person_count = 0
car_count = 0
truck_count = 0
p_arr_all = np.empty((0, 5))
rd = 0
cd = 0
d_weight = np.array([1.25998060e-01, 0.00000000e+00, 3.72307987e-04, 2.97648267e-04,
                     4.07682852e-03, 3.03747762e-03, 2.07011840e-02, 2.03772601e-01,
                     5.00000000e-01])

path = mpltPath.Path(hull)

c_list = np.empty((0,4))
pts_list = []
broken_radar_track = {'case_1': [], 'case_2': [], 'case_3': [], 'case_4': []}
start_list = [-1] * 8000
end_list = [-1] * 8000
prev_track_list = [-1] * 8000
unconfirmed_track = 0
road_objects_c = []
road_objects_r = []
road_objects_dict = dict()
road_objects_dict['no_match'] = [[], [], 0]
radar_id_count = 0
radar_ids = [-1]
footage = 'record/remote'
frame = SRS_data_frame()
with Reader(footage) as reader:
    connections = [x for x in reader.connections if x.topic in ['/Radar', '/Camera']]

    # for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
    #     msg = deserialize_cdr(rawdata, connection.msgtype)
    #     msg.topic = connection.topic

        # init plt figure

        # create generator object for recording
    bg = reader.messages(connections=connections)
    connection, timestamp, rawdata = next(bg)

    def animate(g):
        global image_np, g2c_p, r2c_e, c2g
        global p_arr_all
        global camera_time_0
        global update
        global cam1
        global tracked_object_radar
        global alive_track
        global life
        global t_array
        global dt_array
        global radar_p_time
        global dist
        global mtx
        global roi
        global cam_msg
        global model
        global idx
        global epoch
        global car_count
        global bus_count
        global person_count
        global truck_count
        global hull
        global run_cam_d
        global rd
        global cd
        global outputs
        global d_weight
        global c_list, pts_list
        global broken_radar_track, unconfirmed_track
        global path
        global start_list, prev_track_list, road_objects_c, road_objects_r, road_objects_dict, radar_id_count, radar_ids
        global arr_all, pc, arr_concat, p_arr_all
        global radar_track
        if idx <= 150:
            connection, timestamp, rawdata =  next(bg)            # read ros Topic camera or radar
            idx+=1
        else:
            connection, timestamp, rawdata = next(bg)
            msg = deserialize_cdr(rawdata, connection.msgtype)
            msg.topic = connection.topic
            sensor = frame.load_data(msg)
            if sensor == "/Radar":
                npts = frame.radar.width
                arr_all = pc2_numpy(frame.radar, npts)
                # arr_concat = np.vstack((arr_all, p_arr_all))
                arr_concat = arr_all
                p_arr_all = arr_concat.copy()
            frame_check = frame.full_data
            if frame_check:

                arr_c = transform_radar(arr_concat.T, r2c_e).T  # from radar to camera (not pixel)
                arr_g = transform_radar(arr_c.T, c2g).T  # from camera to global
                mask = path.contains_points(np.vstack((arr_g[:, 0], arr_g[:, 2])).T)

                arr = arr_g[mask]
                pc = arr[:, :4]
                image_np = imgmsg_to_cv2(frame.camera)
                plt.cla()
                axs.set_xlim(-100, 100)
                axs.set_ylim(-100, 100)
                plt.plot(hull[:, 0], hull[:, 1], 'k-')


                # draw points on plt figure

                # Perform class specific DBSCAN
                _, _, C_M, camera_detection, outputs = process_track(image_np, msg, curr_frames, prev_frames,
                                                                         outputs,
                                                                         device, model, stride, names, pt, imgsz, cfg,
                                                                         strongsort_list, dt,
                                                                         seen, half, classes=[2], conf_thres=0.75)
                # [5, 7, 0, 2]
                im0 = C_M.copy()
                cam_2d = np.asarray([ii[0] for ii in camera_detection])
                detection_list, img, measSet = detection_utils.radar_detection(arr, C_M, pc, camera_detection, cam_2d, g2c_p, scaler_model, svm_model )
                # perform tracking
                radar_track.update(measSet)
                detection_list = radar_track.matching(detection_list, img, parameter)

                for ii in detection_list:
                    # check if this camera id then radar id is in any of the tracks
                    in_track = False
                    ids = []
                    if ii.rad_id == 28:
                        print(ii.cam_id)
                    for jj, tk in enumerate(tracked_object_all):
                        # record all track with the same radar or camera ID in ids array
                        if not tk.deleted:

                            if ii.cam_id and ii.cam_id in tk.c_id:
                                in_track = True
                                ids.append(jj)
                                if ii.rad_id and ii.rad_id not in tk.r_id:
                                    tk.r_id.append(ii.rad_id)
                                elif ii.rad_id and ii.rad_id in tk.r_id:
                                    tk.r_id.append(tk.r_id.pop(tk.r_id.index(ii.rad_id)))
                            elif ii.rad_id and ii.rad_id in tk.r_id:
                                in_track = True
                                ids.append(jj)
                                if ii.cam_id and ii.cam_id not in tk.c_id:
                                    tk.c_id.append(ii.cam_id)
                                elif ii.cam_id and ii.cam_id in tk.c_id:
                                    tk.c_id.append(tk.c_id.pop(tk.c_id.index(ii.cam_id)))
                    if in_track:
                        # merge all the track into one track
                        # if camera id is found
                        tracked_object_all[ids[0]].activate = ii.sensor
                        # add the radar id to this tracked object if not there
                    elif ii.cam_id or ii.rad_id:
                        trk = TrackedObjectALL(c_id=ii.cam_id, r_id=ii.rad_id, sensor=ii.sensor)
                        tracked_object_all.append(trk)

                for ii in detection_list:
                    if ii.sensor in ['Both', 'Camera']:
                        box = ii.cam_box
                        py = min([box[1], box[3]])
                        px = (box[0]+box[2])/2
                        g_x, g_z = Cam2Ground(px, py, c2wcs)
                        axs.scatter(-g_z, g_x, s=5)
                font_size = 0.3
                thickness = 1

                for ii, trk in enumerate(tracked_object_all):
                    if True:
                        if not trk.deleted and trk.activate:
                            if trk.activate in ['Radar', 'Both', 'Radar_track'] and trk.r_id:
                                r_id = trk.r_id
                                r_trk = radar_track.trackList[r_id[-1]]
                                centroid = r_trk.xPost[:2]
                                speed = r_trk.speed
                                centroid_img = np.zeros((1, 4))
                                centroid_img[:, 0] = centroid[0]
                                centroid_img[:, 2] = centroid[1]
                                centroid_img[:, 1] = 0
                                pts = project_to_image(centroid_img.T, g2c_p).flatten()
                                cv2.putText(im0, f'Fusion ID: {ii}', (
                                    int(pts[0] + 10), int(pts[1]) - 15),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            font_size, (0, 255, 0), thickness, cv2.LINE_AA)
                                cv2.putText(im0, f'Radar_id: {r_id}', (
                                    int(pts[0] + 10), int(pts[1]) +15),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            font_size, (0, 255, 0), thickness, cv2.LINE_AA)
                                cv2.putText(im0, f'Speed: {speed:.03g}km/h', (
                                    int(pts[0]) + 10, int(pts[1]) + 3),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            font_size, (0, 255, 0), 1, cv2.LINE_AA)
                                cv2.circle(im0, (int(pts[0]), int(pts[1])), 5, color=(0, 255, 0), thickness=2)
                                if trk.c_id:
                                    cv2.putText(im0, f'camera_id: {trk.c_id}', (
                                        int(pts[0] + 10), int(pts[1]) + 25),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                font_size, (0, 255, 0), thickness, cv2.LINE_AA)
                                trk.life = 10
                            elif trk.c_id:
                                c_id = trk.c_id
                                if c_id[-1] in outputs[0][:, 4]:
                                    id= np.where(outputs[0][:, 4] == c_id[-1])[0][0] # what?
                                    bbox = outputs[0][id, :4]
                                    pts = [(bbox[2] - bbox[0]) / 2 + bbox[0], bbox[3] - 10]
                                    if not trk.r_id:
                                        cv2.putText(im0, f'Fusion ID: {ii}', (
                                            int(pts[0] + 10), int(pts[1])),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    font_size, (0, 0, 255), thickness, cv2.LINE_AA)
                                        cv2.circle(im0, (int(pts[0]), int(pts[1])), 5, color=(0, 0, 255), thickness=2)
                                        cv2.putText(im0, f'camera_id: {c_id}', (
                                            int(pts[0] + 10), int(pts[1])+ 13),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    font_size, (0, 0, 255), thickness, cv2.LINE_AA)
                                    if trk.r_id:
                                        r_id = trk.r_id
                                        speed = radar_track.trackList[r_id[-1]].speed
                                        cv2.circle(im0, (int(pts[0]), int(pts[1])), 5, color=(0, 0, 255), thickness=2)
                                        cv2.putText(im0, f'Fusion ID: {ii}', (
                                            int(pts[0] + 10), int(pts[1])- 15),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    font_size, (0, 255, 0), thickness, cv2.LINE_AA)
                                        cv2.putText(im0, f'camera_id: {c_id}', (
                                            int(pts[0] + 10), int(pts[1])+25),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    font_size, (0, 0, 255), thickness, cv2.LINE_AA)
                                        cv2.putText(im0, f'Speed: {speed:.03g}km/h', (
                                            int(pts[0]) + 10, int(pts[1]) + 3),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    font_size, (0, 0, 255), thickness, cv2.LINE_AA)
                                        cv2.putText(im0, f'Radar_id: {r_id} (noD)', (
                                            int(pts[0] + 10), int(pts[1]) + 15),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    font_size, (0, 0, 255), thickness, cv2.LINE_AA)
                                        trk.life = 10
                            trk.activate = False

                im0, cam_arr = render_radar_on_image(arr_g, im0, g2c_p, 9000, 9000)
                cv2.imshow('Camera', im0)
                cv2.waitKey(1)


                idx += 1
                p_arr_all = arr_all.copy()



    ani = FuncAnimation(fig, animate, interval=10, frames=2000)
    plt.show()

