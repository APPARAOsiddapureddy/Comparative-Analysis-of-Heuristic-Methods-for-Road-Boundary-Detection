import math
from ultralytics import YOLO
import cv2
from sort2 import *
from DOM_y import intialize_y_direction
import numpy as np

COCO_ClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                   "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                   "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                   "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                   "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                   "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                   "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                   "teddy bear", "hair drier", "toothbrush"]

tracking_obj2 = {}
l_l_b_points = []
l_r_b_points = []
r_l_b_points = []
r_r_b_points = []

l_l_l_b_Id, l_l_r_b_Id, r_l_l_b_Id, r_l_r_b_Id = intialize_y_direction()


cap2 = cv2.VideoCapture("../Videos/footage_6.mp4")
model2 = YOLO("./YOLOWeights/yolov8n.pt")
tracker2 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

img_copy = []
while True:
    print(l_l_l_b_Id, l_l_r_b_Id)
    print(r_l_l_b_Id, r_l_r_b_Id)
    success, img = cap2.read()
    if not success:
        break

    height = img.shape[0]
    width = img.shape[1]
    img_copy = img.copy()
    # img = cv2.bitwise_and(img, img, mask=total_foreground)
    results = model2(img_copy, stream=True)
    detections = np.empty((0, 5))

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = box.cls[0]
            x, y, x_w, y_h = box.xyxy[0]
            x1, y1, x2, y2 = int(x), int(y), int(x_w), int(y_h)
            conf_val = math.ceil((box.conf[0] * 100)) / 100
            # cv2.putText(img, f'{cls_name}', (x1+10, y1 - 20), cv2.FONT_ITALIC, 0.7, (0, 255, 255), 2)
            current_arr = np.array([x1, y1, x2, y2, conf_val])
            detections = np.vstack((detections, current_arr))

    resultsTracker = tracker2.update(detections)

    for result in resultsTracker:
        x, y, x_w, y_h, Id = result
        x1, y1, x2, y2, Id = int(x), int(y), int(x_w), int(y_h), int(Id)
        cv2.putText(img_copy, f'Id-', (x1+10, y1 - 20), cv2.FONT_ITALIC, 0.55, (255, 255, 0), 2)
        cv2.putText(img_copy, f'{Id}', (x1 + 40, y1 - 20), cv2.FONT_ITALIC, 0.6, (0, 255, 255), 2)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 255), 2)
        w, h = x2-x1, y2 - y1
        cx, cy = x1 + w//2, y1 + h//2

        if l_l_l_b_Id in tracking_obj2:
            # cv2.circle(img, tracking_obj[l_l_l_b_Id], 2, (0, 255, 255), 2)
            prev_pt = tracking_obj2[l_l_l_b_Id]
            l_l_b_points.append((prev_pt[0], prev_pt[1] + prev_pt[3]//2))
            if Id == l_l_l_b_Id:
                tracking_obj2[l_l_l_b_Id] = (x1, y1, w, h)

        if l_l_r_b_Id in tracking_obj2:
            # cv2.circle(img, tracking_obj[l_l_r_b_Id], 2, (0, 255, 255), 2)
            prev_pt = tracking_obj2[l_l_r_b_Id]
            l_r_b_points.append((prev_pt[0]+prev_pt[2], prev_pt[1] + prev_pt[3]//2))
            if Id == l_l_r_b_Id:
                tracking_obj2[l_l_r_b_Id] = (x1, y1, w, h)

        if r_l_l_b_Id in tracking_obj2:
            # cv2.circle(img, tracking_obj[l_l_l_b_Id], 2, (0, 255, 255), 2)
            prev_pt = tracking_obj2[r_l_l_b_Id]
            r_l_b_points.append((prev_pt[0], prev_pt[1] + prev_pt[3]//2))
            if Id == r_l_l_b_Id:
                tracking_obj2[r_l_l_b_Id] = (x1, y1, w, h)

        if r_l_r_b_Id in tracking_obj2:
            # cv2.circle(img, tracking_obj[l_l_r_b_Id], 2, (0, 255, 255), 2)
            prev_pt = tracking_obj2[r_l_r_b_Id]
            r_r_b_points.append((prev_pt[0]+prev_pt[2], prev_pt[1] + prev_pt[3]//2))
            if Id == r_l_r_b_Id:
                tracking_obj2[r_l_r_b_Id] = (x1, y1, w, h)

        tracking_obj2[Id] = (x1, y1, w, h)
        if len(l_l_b_points) != 0:
            for i in range(len(l_l_b_points) - 1):
                cv2.line(img_copy, l_l_b_points[i], l_l_b_points[i + 1], (0, 255, 0), 2)

        if len(l_r_b_points) != 0:
            for i in range(len(l_r_b_points) - 1):
                cv2.line(img_copy, l_r_b_points[i], l_r_b_points[i + 1], (0, 255, 0), 2)

        if len(r_l_b_points) != 0:
            for i in range(len(r_l_b_points) - 1):
                cv2.line(img_copy, r_l_b_points[i], r_l_b_points[i + 1], (255, 255, 0), 2)

        if len(r_r_b_points) != 0:
            for i in range(len(r_r_b_points) - 1):
                cv2.line(img_copy, r_r_b_points[i], r_r_b_points[i + 1], (255, 255, 0), 2)

    cv2.imshow("Video", img_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

l_l_b_points_np = np.array(l_l_b_points)
l_l_b_points_avg_x = np.mean(l_l_b_points_np[:, 0])
l_l_b_points_avg_y = np.mean(l_l_b_points_np[:, 1])
print("llb:", (l_l_b_points_avg_x, l_l_b_points_avg_y))

l_r_b_points_np = np.array(l_r_b_points)
l_r_b_points_avg_x = np.mean(l_r_b_points_np[:, 0])
l_r_b_points_avg_y = np.mean(l_r_b_points_np[:, 1])
print("lrb:", (l_r_b_points_avg_x, l_r_b_points_avg_y))

r_l_b_points_np = np.array(r_l_b_points)
r_l_b_points_avg_x = np.mean(r_l_b_points_np[:, 0])
r_l_b_points_avg_y = np.mean(r_l_b_points_np[:, 1])
print("rlb:", (r_l_b_points_avg_x, r_l_b_points_avg_y))

r_r_b_points_np = np.array(r_r_b_points)
r_r_b_points_avg_x = np.mean(r_r_b_points_np[:, 0])
r_r_b_points_avg_y = np.mean(r_r_b_points_np[:, 1])
print("rrb:", (r_r_b_points_avg_x, r_r_b_points_avg_y))


cap2.release()
cv2.destroyAllWindows()
