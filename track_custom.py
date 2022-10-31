import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
print('track')
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'strong_sort/deep/reid') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort/deep/reid'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])


@torch.no_grad()
def load_weight_sort(device, config_file, half=False, yolo_weights=WEIGHTS / 'yolov5m.pt', imgsz=(1280, 720),
                strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt', nr_sources=1):
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=False, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    cfg = get_config()
    print(config_file)
    cfg.merge_from_file(config_file)
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    return device, model, stride, names, pt, imgsz, cfg, strongsort_list, dt, seen, curr_frames, prev_frames, half


@torch.no_grad()
def process_track(im, idx, curr_frames, prev_frames, outputs,
                device, model, stride, names, pt, imgsz, cfg, strongsort_list, dt, seen, half,
                conf_thres = 0.75,  # confidence threshold
                iou_thres = 0.45,  # NMS IOU threshold
                max_det = 1000,  # maximum detections per image
                classes = None,
                agnostic_nms = False,  # class-agnostic NMS
                  ):
    im0s = im.copy()
    im = letterbox(im, 1280, stride=32, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    pred = model(im, augment=False, visualize=False)
    t3 = time_sync()
    dt[1] += t3 - t2

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3
    # Process detections
    dets = []
    for i, det in enumerate(pred):  # detections per image
        seen += 1
        im0 =  im0s.copy()
        curr_frames[i] = im0
        annotator = Annotator(im0, line_width=2, pil=not ascii)
        if cfg.STRONGSORT.ECC:  # camera motion compensation
            strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]

            # pass detections to strongsort
            t4 = time_sync()
            outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
            t5 = time_sync()
            dt[3] += t5 - t4
            # draw boxes for visualization
            dets = []
            if len(outputs[i]) > 0:
                for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]
                    c = int(cls)  # integer class
                    id = int(id)  # integer id
                    label = f'{id} {conf:.2f} {names[c]}'
                    # label = f'{id}'

                    dets.append([bboxes, names[c], id, float(conf)])
                    annotator.box_label(bboxes, label, color=colors(c, True))
                    im0 = annotator.result()
            # LOGGER.info(f'{idx}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

        else:
            strongsort_list[i].increment_ages()
            # LOGGER.info('No detections')

        # Stream results
        im0 = annotator.result()
        prev_frames[i] = curr_frames[i]
        return prev_frames, curr_frames, im0, dets, outputs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


if __name__ == "__main__":
    from yolov5.utils.augmentations import letterbox
    # opt = parse_opt()

    device = '0'
    device, model, stride, names, pt, imgsz, cfg, strongsort_list, \
    dt, seen, curr_frames, prev_frames, half = load_weight_sort(device, 'strong_sort/configs/strong_sort.yaml')

    outputs=[None]
    for i in range(1340):
        im0s = cv2.imread(f"data/{i:05d}/camera.png")
        _, _, im0, detection = process_track(im0s, i, curr_frames, prev_frames, outputs,
                device, model, stride, names, pt, imgsz, cfg, strongsort_list, dt, seen, half)
        cv2.imshow('0', im0)
        print(detection)
        cv2.waitKey(1)
