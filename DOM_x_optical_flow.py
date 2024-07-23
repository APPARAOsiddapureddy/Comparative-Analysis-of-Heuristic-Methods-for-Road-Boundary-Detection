import numpy as np
import cv2
import time
from RoadSurface_Extraction import extract_road_region

# Parameters for Lucas-Kanade optical flow and feature detection
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=30,
                      qualityLevel=0.3,
                      minDistance=5,
                      blockSize=7)

trajectory_len = 100
detect_interval = 5
trajectories = []
frame_idx = 0
direction_trajectories = {
    'top': [],
    'down': [],
    'left': [],
    'right': [],
    'other': [],
}

final_op = {
    'top': [],
    'down': [],
    'left': [],
    'right': [],
    'other': [],
}

left_limit1 = float('-1')
right_limit1 = float('-1')
left_limit_trajectory1 = []
right_limit_trajectory1 = []

left_limit2 = float('-1')
right_limit2 = float('-1')
left_limit_trajectory2 = []
right_limit_trajectory2 = []

video_path = "footage_4.mp4"

cap = cv2.VideoCapture(video_path)

while True:
    start = time.time()

    suc, frame = cap.read()
    if not suc:
        break
    height = frame.shape[0]
    width = frame.shape[1]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()

    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        new_trajectories = []
        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            if np.linalg.norm(np.array(trajectory[-1]) - np.array((x, y))) < 1.7:
                continue

            trajectory.append((x, y))

            if len(trajectory) > trajectory_len:
                del trajectory[0]

            if trajectory[0][1] < height - 100:
                new_trajectories.append(trajectory)

            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajectories = new_trajectories

        for trajectory in trajectories:
            if len(trajectory) >= 2:
                x1, y1 = trajectory[-2]
                x2, y2 = trajectory[-1]
                direction_vector = np.array([x2 - x1, y2 - y1])
                if np.linalg.norm(direction_vector) > 0:
                    direction_vector = direction_vector / np.linalg.norm(direction_vector)
                    angle = np.arctan2(direction_vector[1], direction_vector[0])
                    hue = int((angle + np.pi) * 90 / np.pi)

                    if 17 <= hue <= 27:
                        direction_trajectories['top'].append(trajectory)
                    elif 153 <= hue <= 168:
                        direction_trajectories['down'].append(trajectory)
                    elif 85 <= hue <= 97:
                        direction_trajectories['right'].append(trajectory)
                    elif 172 <= hue <= 179:
                        direction_trajectories['left'].append(trajectory)
                    else:
                        direction_trajectories['other'].append(trajectory)

                    color = tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]))
                    cv2.polylines(img, [np.int32(trajectory)], False, color, thickness=2)

        for trajectory in direction_trajectories['left']:
            for pt in trajectory:
                if left_limit1 == float('-1'):
                    left_limit1 = pt[1]
                    left_limit_trajectory1 = trajectory
                else:
                    if pt[1] <= left_limit1:
                        left_limit1 = pt[1]
                        left_limit_trajectory1 = trajectory

                if right_limit1 == float('-1'):
                    right_limit1 = pt[1]
                    right_limit_trajectory1 = trajectory
                else:
                    if pt[1] >= right_limit1:
                        right_limit1 = pt[1]
                        right_limit_trajectory1 = trajectory

        cv2.polylines(img, [np.int32(left_limit_trajectory1)], False, (0, 0, 0), thickness=2)
        cv2.polylines(img, [np.int32(right_limit_trajectory1)], False, (0, 0, 0), thickness=2)

        for trajectory in direction_trajectories['right']:
            for pt in trajectory:
                if left_limit2 == float('-1'):
                    left_limit2 = pt[1]
                    left_limit_trajectory2 = trajectory
                else:
                    if pt[1] <= left_limit2:
                        left_limit2 = pt[1]
                        left_limit_trajectory2 = trajectory

                if right_limit2 == float('-1'):
                    right_limit2 = pt[1]
                    right_limit_trajectory2 = trajectory
                else:
                    if pt[1] >= right_limit2:
                        right_limit2 = pt[1]
                        right_limit_trajectory2 = trajectory

        cv2.polylines(img, [np.int32(left_limit_trajectory2)], False, (0, 0, 255), thickness=2)
        cv2.polylines(img, [np.int32(right_limit_trajectory2)], False, (0, 0, 255), thickness=2)

    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    direction_trajectories['top'] = []
    direction_trajectories['down'] = []
    direction_trajectories['left'] = []
    direction_trajectories['right'] = []

    cv2.imshow('Optical Flow', img)

    frame_idx += 1
    prev_gray = frame_gray

    end = time.time()

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
