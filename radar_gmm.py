import mayavi.mlab as mlab
import rospy
import rosbag
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Int32, String
import numpy as np
import cv2
from projectutils import draw_radar
from retina_view.msg import MsgRadarPoint
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from utils import *
from sklearn import mixture
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score
# Press the green button in the gutter to run the script.


class new_clustering:
    def __init__(self):
        # Clutter, pedestrian and car
        self.clf = mixture.GaussianMixture(n_components=3, covariance_type='full')

    def cluster_frame(self, frame_data):
        self.frame_data = np.array(frame_data)
        print(self.frame_data.shape)
        self.clf.fit(self.frame_data)
        self.predict_label = self.clf.predict(self.frame_data)
        # Replace the target class with the predicted label
        self.frame_data[:, 2] = self.predict_label
        return self.frame_data

    def preprocess(self, data):
        processed_data = []
        num_points = 0
        for frame in data:
            processed_frame = []
            num_points = num_points + len(frame)
            for point in frame:
                # Get the point information in Cartesian coord.
                pointR = point[11]
                pointAZ = point[12]
                pointEL = point[13]
                pointX = pointR * np.cos(pointEL) * np.sin(pointAZ)
                pointY = pointR * np.cos(pointEL) * np.cos(pointAZ)
                pointZ = pointR * np.sin(pointEL)
                pointD = point[14]
                pointSNR = point[15]
                pointNoise = point[16]
                # Get the centorid information in Cartesian coord.
                targetX = point[4]
                targetY = point[5]
                targetZ = point[6]
                targetVx = point[7]
                targetVy = point[8]
                targetVz = point[9]
                # Get the point feature vector
                delta_x = pointX - targetX
                delta_y = pointY - targetY
                delta_z = pointZ - targetZ
                delta_D = pointD - (pointX * targetVx + pointY * targetVy + pointZ * targetVz) / pointR
                pointRCS = 4 * 10 * np.log10(pointR) + pointSNR * 0.1 + pointNoise * 0.1
                processed_frame.append([delta_x, delta_y, delta_z, delta_D, pointRCS])
            processed_data.append(processed_frame)
        return len(processed_data), num_points, processed_data

    def fit_GMM(self, training_data):
        frame_num, point_num, preprocessed_training_data = self.preprocess(training_data)
        print("Total training radar frames: %d" % (frame_num))
        print("Total training radar points: %d" % (point_num))
        assert len(training_data) == len(preprocessed_training_data), "ERROR!"
        # Flatten all the data
        preprocessed_training_data_flatten = []
        for frame in preprocessed_training_data:
            preprocessed_training_data_flatten.extend(frame)
        # Convert to numpy array
        preprocessed_training_data_flatten_array = np.array(preprocessed_training_data_flatten)
        # Fit the GMM model using the training dataset
        self.clf.fit(preprocessed_training_data_flatten_array)

    def predict(self, testing_data):
        frame_num, point_num, preprocessed_testing_data = self.preprocess(testing_data)
        print("Total testing radar frames: %d" % (frame_num))
        print("Total testing radar points: %d" % (point_num))
        assert len(testing_data) == len(preprocessed_testing_data), "ERROR!"
        for frame_idx in range(len(testing_data)):
            testing_data_array = np.array(preprocessed_testing_data[frame_idx])
            # Replace the target class with the predicted label
            prediction_array = np.array(testing_data[frame_idx])
            prediction_array[:, 3] = self.clf.predict(testing_data_array)
            testing_data[frame_idx] = list(prediction_array)

    def verify(self, testing_data):
        total_testing_frame_data = []
        for frame in testing_data:
            total_testing_frame_data.extend(frame)
        total_testing_frame_data_array = np.array(total_testing_frame_data)
        print("Total testing data shape is: %s" % (str(total_testing_frame_data_array.shape)))
        # Get the ground truth
        ground_truth = total_testing_frame_data_array[:, 2]
        # Get the prediction
        prediction = total_testing_frame_data_array[:, 3]
        # Calculate the confusin matirx
        obj_class = ['clutter', 'car', 'pedestrian']
        print("***********************************************************************************")
        print(classification_report(ground_truth, prediction, target_names=obj_class))
        print("***********************************************************************************")
        print("IoU report:")
        print(obj_class)
        print("***********************************************************************************")
        self.plot_confusion_matrix(ground_truth, prediction, obj_class, normalize=True)

    def plot_confusion_matrix(self, y_true, y_pred, classes,
                              normalize=True,
                              title=None,
                              cmap=plt.cm.Blues):
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        # classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        plt.show()


def filter_cluster(pc, label):
    msk = label != -1
    pc = pc[msk, :]
    if pc.shape[0] == 0:
        return np.zeros([1,5])

    # print(msk)
    return pc


def set_cluster(pc, label):
    l = np.unique(label)
    pc_list = []
    for i in l:
        if i != -1:
            msk = label == i
            pts = pc[msk, :]
            pc_list.append(pts)
    if not pc_list:
        return [np.zeros([1,5])]
    return pc_list


def convert_to_numpy(pc):
    l = len(pc)
    arr = np.zeros((l, 5))
    for i, point in enumerate(pc):
        arr[i, 0] = point.x
        arr[i, 1] = point.y
        arr[i, 2] = point.z
        arr[i, 3] = point.doppler
    return arr


def filter_zero(pc):
    mask = np.abs(pc[:, 3]) > 0.05
    s = np.sum(mask)
    # print(pc.shape)
    # print(mask)
    # print(mask.shape)
    pc = pc[mask, :]
    if pc.shape[0] == 0:
        return np.zeros([1,5])
    return pc


bag = rosbag.Bag("car_1_ped_1.bag")
topics = bag.get_type_and_topic_info()
# print(topics)

for i in topics[1]:
    print(i)
# print(type(bag))
fig = mlab.figure(size=(1000, 1000), bgcolor=(0, 0, 0))
radar_d = '/radar_data'
view=(180.0, 70.0, 250.0, ([12.0909996 , -1.04700089, -2.03249991]))
image_np = np.empty(0)
for j, i in enumerate(bag.read_messages()):

    if j < 350:
        continue
    if i.topic == '/usb_cam/image_raw/compressed':
        np_arr = np.frombuffer(i.message.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        cv2.imshow('0', image_np)
        cv2.waitKey(1)
    elif i.topic == '/radar_data':
        # print(type(i.message.points[0]))
        # print(i.message.points[0])
        arr = convert_to_numpy(i.message.points)
        arr = filter_zero(arr)

        if image_np.any():
            img, arr = render_lidar_on_image(arr, image_np, np.pi/2, 0, 0, 0, 0, 0, 9000, 9000)
            print(img.shape)
            draw_radar(arr, fig=fig)

        mlab.clf(fig)
        pc = arr[:, :4]

        # pc = StandardScaler().fit_transform(pc)
        clustering = DBSCAN(eps=2, min_samples=40)
        clustering.fit(pc)
        label = clustering.labels_
        cls = set_cluster(pc, label)


        # print(cls)
        # pc = filter_cluster(pc, label)
        # draw_radar(arr, fig=fig)
        # for c in cls:
        #     draw_radar(c, fig=fig, pts_color=(np.random.random(), np.random.random(), np.random.random()),
        #                          view=view)
        # view = mlab.view()

        # mlab.show(stop=True)

    # input()
