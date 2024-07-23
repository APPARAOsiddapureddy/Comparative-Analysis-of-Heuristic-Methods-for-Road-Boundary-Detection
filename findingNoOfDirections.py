import cv2
import numpy as np
from sort import Sort  # Ensure the SORT library is available

class VehicleTracker:
    def __init__(self, threshold=10):
        self.tracker = Sort()
        self.vehicle_dict = {}
        self.threshold = threshold

    def update(self, detections):
        if detections:
            detections = np.array(detections)
            tracked_objects = self.tracker.update(detections)
            current_ids = set()
            for obj in tracked_objects:
                obj_id = int(obj[4])
                bbox = obj[:4]
                current_ids.add(obj_id)
                if obj_id in self.vehicle_dict:
                    self.vehicle_dict[obj_id]['bbox'] = bbox
                    self.vehicle_dict[obj_id]['path'].append(bbox)
                else:
                    self.vehicle_dict[obj_id] = {'id': obj_id, 'bbox': bbox, 'path': [bbox]}
            obsolete_ids = set(self.vehicle_dict.keys()) - current_ids
            for obj_id in obsolete_ids:
                del self.vehicle_dict[obj_id]

    def draw(self, frame):
        for vehicle in self.vehicle_dict.values():
            x1, y1, x2, y2 = vehicle['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {vehicle['id']}, Coords: ({x1},{y1})-({x2},{y2})",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def get_vehicle_directions(self):
        directions = {}
        for vehicle in self.vehicle_dict.values():
            path = vehicle['path']
            if len(path) < 2:
                continue
            
            x_diff = path[-1][0] - path[0][0]
            y_diff = path[-1][1] - path[0][1]

            if abs(x_diff) > abs(y_diff):
                if x_diff > 0:
                    direction = 'right'
                else:
                    direction = 'left'
            else:
                if y_diff > 0:
                    direction = 'down'
                else:
                    direction = 'up'

            directions[vehicle['id']] = direction
        
        return directions

def find_vehicle_boundaries(video_path, max_frames=250):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
    tracker = VehicleTracker()
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        height, width = frame.shape[:2]
        roi_frame = frame[int(height * 0.4):, :]

        fg_mask = bg_subtractor.apply(roi_frame)
        fg_mask = cv2.threshold(fg_mask, 230, 255, cv2.THRESH_BINARY)[1]
        fg_mask = cv2.erode(fg_mask, None, iterations=3)
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x += 0
            y += int(height * 0.4)
            if w > 30 and h > 30:
                detections.append([x, y, x + w, y + h, 1])

        tracker.update(detections)
        tracker.draw(frame)

        # Remove or comment out this line to not show the video
        # cv2.imshow("Vehicle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # cv2.destroyAllWindows()

    directions = tracker.get_vehicle_directions()
    return directions

if __name__ == "__main__":
    video_path = "/mnt/data/footage_6.mp4"  # Correct path to the uploaded video
    max_frames = 250
    directions = find_vehicle_boundaries(video_path, max_frames)
    print("Vehicle Directions:", directions)
