import time
import cv2
import numpy as np
import os
from sort import Sort

class VehicleTracker:
    def __init__(self):
        self.tracker = Sort()
        self.vehicle_dict = {}
        self.boundaries_top = []
        self.boundaries_bottom = []

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
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {vehicle['id']}, Coordinates: ({x1},{y1}) - ({x2},{y2})",
                        (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            path = vehicle['path']
            if len(path) > 1:
                for i in range(1, len(path)):
                    start_point1 = (int((path[i-1][0] + path[i-1][2]) / 2), int(path[i-1][1]))
                    end_point1 = (int((path[i][0] + path[i][2]) / 2), int(path[i][1]))
                    start_point2 = (int((path[i-1][0] + path[i-1][2]) / 2), int(path[i-1][3]))
                    end_point2 = (int((path[i][0] + path[i][2]) / 2), int(path[i][3]))
                    cv2.line(frame, start_point1, end_point1, (0, 0, 255), 2)
                    cv2.line(frame, start_point2, end_point2, (0, 0, 255), 2)
                    self.boundaries_top.append((start_point1, end_point1))
                    self.boundaries_bottom.append((start_point2, end_point2))

    def draw_static_boundaries(self, frame):
        for boundary in self.boundaries_top:
            cv2.line(frame, boundary[0], boundary[1], (255, 0, 0), 2)
        for boundary in self.boundaries_bottom:
            cv2.line(frame, boundary[0], boundary[1], (0, 0, 255), 2)

def find_vehicle_boundaries(image_folder, max_frames=200):
    image_files = sorted([img for img in os.listdir(image_folder) if img.endswith('.jpg') or img.endswith('.png')])
    if len(image_files) < max_frames:
        print(f"Warning: Found only {len(image_files)} images, less than the requested {max_frames}.")
    image_files = image_files[:max_frames]

    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    tracker = VehicleTracker()
    frame_count = 0

    for image_file in image_files:
        frame = cv2.imread(os.path.join(image_folder, image_file))
        if frame is None:
            print(f"Error: Could not read image {image_file}.")
            continue

        frame_count += 1

        width = frame.shape[1]
        height = frame.shape[0]

        # ROI dimensions (adjust as needed)
        roi_x = int(width * 0)
        roi_y = int(height * 0.4)
        roi_width = int(width * 1)
        roi_height = int(height * 0.6)

        roi_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        fg_mask = bg_subtractor.apply(roi_frame)
        fg_mask = cv2.threshold(fg_mask, 230, 255, cv2.THRESH_BINARY)[1]
        fg_mask = cv2.erode(fg_mask, None, iterations=3)
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Adjust the bounding box coordinates based on the ROI
            x += roi_x
            y += roi_y
            width_cm = (w / width) * 100
            height_cm = (h / height) * 100
            if height_cm < 4:
                continue
            if width_cm < 4:
                continue
            detections.append([x, y, x + w, y + h, 1])

        tracker.update(detections)

    return tracker.boundaries_top, tracker.boundaries_bottom

def mark_vehicle_boundaries(image_folder, min_max_boundaries, output_path, max_frames=200):
    image_files = sorted([img for img in os.listdir(image_folder) if img.endswith('.jpg') or img.endswith('.png')])
    if len(image_files) < max_frames:
        print(f"Warning: Found only {len(image_files)} images, less than the requested {max_frames}.")
    image_files = image_files[:max_frames]

    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    tracker = VehicleTracker()
    frame_count = 0

    # Video writer setup
    first_frame = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    for image_file in image_files:
        frame = cv2.imread(os.path.join(image_folder, image_file))
        if frame is None:
            print(f"Error: Could not read image {image_file}.")
            continue

        frame_count += 1

        width = frame.shape[1]
        height = frame.shape[0]

        # ROI dimensions (adjust as needed)
        roi_x = int(width * 0)
        roi_y = int(height * 0.4)
        roi_width = int(width * 1)
        roi_height = int(height * 0.6)

        roi_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        fg_mask = bg_subtractor.apply(roi_frame)
        fg_mask = cv2.threshold(fg_mask, 230, 255, cv2.THRESH_BINARY)[1]
        fg_mask = cv2.erode(fg_mask, None, iterations=3)
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Adjust the bounding box coordinates based on the ROI
            x += roi_x
            y += roi_y
            width_cm = (w / width) * 100
            height_cm = (h / height) * 100
            if height_cm < 3:
                continue
            if width_cm < 5:
                continue
            detections.append([x, y, x + w, y + h, 1])

        tracker.update(detections)
        tracker.draw(frame)
        tracker.draw_static_boundaries(frame)  # Draw static boundaries

        # Display the frame
        cv2.imshow('Frame', frame)

        # Write the frame to the output video file
        out.write(frame)

        # Add a delay (e.g., 30 milliseconds)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()

    print("Final output video saved as:", output_path)

# Example usage:
image_folder = 'MVI_39031'
output_path = 'final_output.avi'

start_time = time.time()
min_max_boundaries = find_vehicle_boundaries(image_folder)
end_time = time.time()
print(f"Time taken to find vehicle boundaries: {end_time - start_time:.2f} seconds")

start_time = time.time()
mark_vehicle_boundaries(image_folder, min_max_boundaries, output_path)
end_time = time.time()
print(f"Time taken to mark vehicle boundaries: {end_time - start_time:.2f} seconds")
