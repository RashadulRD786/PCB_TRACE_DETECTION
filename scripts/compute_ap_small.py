import os
import numpy as np

IMG_SIZE = 600
SMALL_THRESH = 32 * 32
IOU_THRESH = 0.5

gt_dir = "../datasets/HRIPCB/val/labels"
# pred_dir = "runs/detect/predict/labels"
pred_dir = "runs/detect/runs/detect/predict/yolov8s_960_100/labels"

def yolo_to_bbox(x, y, w, h):
    w *= IMG_SIZE
    h *= IMG_SIZE
    x *= IMG_SIZE
    y *= IMG_SIZE

    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2

    return np.array([x1, y1, x2, y2])

def bbox_area(box):
    w = box[2] - box[0]
    h = box[3] - box[1]
    return w * h

def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB-xA) * max(0, yB-yA)

    areaA = bbox_area(boxA)
    areaB = bbox_area(boxB)

    union = areaA + areaB - inter

    return inter / union if union > 0 else 0

gt_boxes = []
pred_boxes = []

for file in os.listdir(gt_dir):

    gt_file = os.path.join(gt_dir, file)
    pred_file = os.path.join(pred_dir, file)

    if not os.path.exists(pred_file):
        continue

    with open(gt_file) as f:
        for line in f:
            c,x,y,w,h = map(float,line.split())
            box = yolo_to_bbox(x,y,w,h)

            if bbox_area(box) < SMALL_THRESH:
                gt_boxes.append((file,box))

    with open(pred_file) as f:
        for line in f:
            c,x,y,w,h,conf = map(float,line.split())
            box = yolo_to_bbox(x,y,w,h)
            pred_boxes.append((file,box,conf))

pred_boxes = sorted(pred_boxes,key=lambda x:x[2],reverse=True)

tp = []
fp = []
matched = set()

for img,box,conf in pred_boxes:

    best_iou = 0
    best_idx = -1

    for i,(gt_img,gt_box) in enumerate(gt_boxes):

        if gt_img != img or i in matched:
            continue

        val = iou(box,gt_box)

        if val > best_iou:
            best_iou = val
            best_idx = i

    if best_iou >= IOU_THRESH:
        tp.append(1)
        fp.append(0)
        matched.add(best_idx)
    else:
        tp.append(0)
        fp.append(1)

tp = np.cumsum(tp)
fp = np.cumsum(fp)

recall = tp / len(gt_boxes)
precision = tp / (tp + fp + 1e-6)

ap = 0
for i in range(1,len(recall)):
    ap += (recall[i]-recall[i-1]) * precision[i]

print("\nAP_small:", round(ap,4))