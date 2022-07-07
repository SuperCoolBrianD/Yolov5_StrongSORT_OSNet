from radar_utils import *
import rosbag





if __name__ == "__main__":
    points = np.array([
        [0.1,0.1,0.1],
        [0.5,0.5,0.5],
        [1.7,1.7,1.7],
        [1.8,1.8,1.8],
        [9.3,9.4,9.4]])
    voxel_size=6
    nb_vox=np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)

    bag = rosbag.Bag("record/car.bag")

    # print(data)
    all_bb = []
    idx = 0
    missidx = 0
    mtx = np.array([[748, 0., 655.5],
                    [0., 746.6, 390.11],
                    [0., 0., 1.]])
    rx = 1.56
    ry = 0
    rz = -.02
    tx = 0
    ty = 0
    tz = 0.1

    r2c = cam_radar(rx, ry, rz, tx, ty, tz, mtx)
    frame = SRS_data_frame()
    epoch = None
    time_arr = []
    voxel_size = [0.05, 0.05, 8.0]
    grid_meters = [154, 154.0, 15]
    bbox_voxel_size = [0.25, 0.25, 1.0]
    label = \
        [[5.309664011001587, 19.599610328674316, 0.25392503291368484, 8, 8, 8, 0],
         [20, 30, 0.25392503291368484, 8, 8, 8, 0],
         [50, 80, 0.25392503291368484, 8, 8, 8, 0],
         [60, 15, 0.25392503291368484, 8, 8, 8, 0],
         [40, 40, 0.25392503291368484, 8, 8, 8, 0],
         [5.309664011001587, 19.599610328674316, 0.25392503291368484, 8, 8, 8, 0],
         [5.309664011001587, 19.599610328674316, 0.25392503291368484, 8, 8, 8, 0],
         ]
    l = make_eight_points_boxes(label)
    bbox = get_bboxes_grid(np.array([255]*7), l[0], grid_meters, bbox_voxel_size)
    for i in range(9):
        img = bbox[:, :, i]
        ret, img = cv2.threshold(img, 1, 100, 0)
        cv2.imshow('0', img)
        print(i)
        cv2.waitKey(1000)
    print(bbox.shape)






    # for j, i in enumerate(bag.read_messages()):
    #     # read ros Topic camera or radar
    #     sensor = frame.load_data(i)
    #     # print(idx)
    #     # print(sensor)
    #     if frame.full_data:
    #         time_arr.append(frame.camera.message.header.stamp.to_sec() - frame.radar.message.header.stamp.to_sec())
    #         # print(frame.camera.message.header.stamp.to_sec()- epoch)
    #         image_np = imgmsg_to_cv2(frame.camera.message)
    #         npts = frame.radar.message.width
    #         arr_all = pc2_numpy(frame.radar.message, npts)[:, :4]
    #         arr_all[:, 0] += 50
    #         arr_all[:, 1] += 50
    #         arr_all[:, 2] += 2
    #         top_view = make_top_view_image(arr_all, grid_meters, voxel_size)
    #         resized = cv2.resize(top_view[:, :, 0], (770, 770), interpolation=cv2.INTER_AREA)
    #         cv2.imshow('0', resized)
    #         cv2.waitKey(50)
    #         # print(arr_all)
    #         # print(top_view.shape)
    #         # draw points on plt figure
    #         # arr = filter_zero(arr_all)
    #         # total_box, cls = dbscan_cluster(arr, eps=2, min_sample=20)



    # plt.hist(time_arr, bins=50)
    # print(sum(time_arr)/len(time_arr))
    # # 1.08
    # plt.show()