import numpy as np
import cv2

def IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def find_gt_viz(r_box, c_box, image=np.empty([])):
    """
    Parameters
    ----------
    box1 : TYPE
        DESCRIPTION.
    gt : list containing all 2D detection

    Returns
    -------
    i : TYPE
        DESCRIPTION.

    """
    '''
    Match radar and camera detection using 2D boxes
    '''
    matched = [[], 'no_match', 0]
    max_iou = 0
    w = 1
    s = 1
    # if image.any():
    #     img = cv2.rectangle(image.copy(), (r_box[0], r_box[1]), (r_box[2], r_box[3]), (0, 255, 0), thickness=2)
    #     cv2.putText(img, f'Radar Cluster', (r_box[0], r_box[1]),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    for i in c_box:
        box = [int(ii) for ii in i[0]]
        iou = IOU(r_box, box)
        # if image.any():
        #     img1 = cv2.rectangle(img.copy(), (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), thickness=2)
        #     cv2.putText(img1, f'YOLOR Detection: {i[1]} iou: {iou}', (i[0][0]-15, i[0][1]-15),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.imshow('camera', img1)
            # cv2.waitKey(w)

        if iou > max_iou:
            matched = i
            max_iou = iou

    # if image.any() and matched[0]:
    #     img1 = cv2.rectangle(img.copy(), (matched[0][0], matched[0][1]), (matched[0][2], matched[0][3]), (255, 255, 255), thickness=2)
    #     cv2.putText(img1, f'Matched: {matched[1]}, iou: {max_iou}', (matched[0][0] - 15, matched[0][1] - 15),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.imshow('camera', img1)
        # cv2.waitKey(w+s)
    return matched, max_iou



def find_gt(r_box, c_box, image=np.empty([])):
    """
    Parameters
    ----------
    box1 : TYPE
        DESCRIPTION.
    gt : list containing all 2D detection

    Returns
    -------
    i : TYPE
        DESCRIPTION.

    """
    '''
    Match radar and camera detection using 2D boxes
    '''
    matched = [[], 'no_match', 0]
    max_iou = 0
    for i in c_box:
        box = [int(ii) for ii in i[0]]
        iou = IOU(r_box, box)
        if iou > max_iou:
            matched = i
            max_iou = iou
    return matched, max_iou


def match_detection(r_box, c_box):
    iou_arr = np.zeros((r_box.shape[0], c_box.shape[0]))
    for i in range(r_box.shape[0]):
        for j in range(c_box.shape[0]):
            c = c_box[j, :]
            r = r_box[i, :]
            iou = IOU(r, c)
            iou_arr[i, j] = iou
    radar_matched = []
    camera_matched = []
    ious = []
    for i in range(iou_arr.shape[0]):
        picked = np.argmax(iou_arr)
        picked = np.unravel_index(picked, iou_arr.shape)
        iou = iou_arr[picked[0], picked[1]]
        if iou == 0:
            break
        ious.append(iou)
        radar_matched.append(picked[0])
        camera_matched.append(picked[1])
        iou_arr[picked[0], :] = -1
        iou_arr[:, picked[1]] = -1
    radar_unmatched = []
    for i in range(iou_arr.shape[0]):
        if i not in radar_matched:
            radar_unmatched.append(i)
    camera_unmatched = []
    for i in range(iou_arr.shape[1]):
        if i not in camera_matched:
            camera_unmatched.append(i)
    return radar_matched, camera_matched, ious, radar_unmatched, camera_unmatched


def np2bin(pc):
    """Convert Numpy format pointcloud to KITTI style"""
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    doppler = pc[:, 3]
    arr = np.zeros(x.shape[0] + y.shape[0] + z.shape[0] + doppler.shape[0], dtype=np.float32)
    arr[::4] = x
    arr[1::4] = y
    arr[2::4] = z
    arr[3::4] = doppler
    return arr


def convert_xyxy(box):
    return (box[0], box[1]), (box[2], box[3])


def convert_topy_bottomx(box):
    topx = min(box[0], box[2])
    topy = min(box[1], box[3])
    bottomx = max(box[0], box[2])
    bottomy = max(box[1], box[3])
    return [topx, topy, bottomx, bottomy]