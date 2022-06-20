
from radar_utils import *
import rosbag


def get_voxels_grid(voxel_size, grid_meters):
    voxel_size = np.asarray(voxel_size, np.float32)
    grid_size_meters = np.asarray(grid_meters, np.float32)
    voxels_grid = np.asarray(grid_size_meters / voxel_size, np.int32)
    return voxels_grid


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
    for j, i in enumerate(bag.read_messages()):
        # read ros Topic camera or radar
        sensor = frame.load_data(i)
        # print(idx)
        # print(sensor)
        if frame.full_data:
            time_arr.append(frame.camera.message.header.stamp.to_sec() - frame.radar.message.header.stamp.to_sec())
            # print(frame.camera.message.header.stamp.to_sec()- epoch)
            image_np = imgmsg_to_cv2(frame.camera.message)
            npts = frame.radar.message.width
            arr_all = pc2_numpy(frame.radar.message, npts)[:, :4]
            arr_all[:, 0] += 50
            arr_all[:, 1] += 50
            arr_all[:, 2] += 2
            top_view = make_top_view_image(arr_all, grid_meters, voxel_size)
            resized = cv2.resize(top_view[:, :, 0], (770, 770), interpolation=cv2.INTER_AREA)
            cv2.imshow('0', resized)
            cv2.waitKey(50)
            # print(arr_all)
            # print(top_view.shape)
            # draw points on plt figure
            # arr = filter_zero(arr_all)
            # total_box, cls = dbscan_cluster(arr, eps=2, min_sample=20)



    plt.hist(time_arr, bins=50)
    print(sum(time_arr)/len(time_arr))
    # 1.08
    plt.show()