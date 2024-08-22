import cv2
import numpy as np
import random
import colorsys
import argparse

class ObjectTracker:
    def __init__(self, max_frames_to_skip=50, disappear_prob=0.025, max_distance=100):
        self.objects = {}
        self.next_id = 0
        self.max_frames_to_skip = max_frames_to_skip
        self.max_distance = max_distance
        self.disappear_prob = disappear_prob
        self.used_colors = set()
        self.min_size = 10

    def update(self, detections, frame):
        updated_ids = []
        for (x, y, w, h) in detections:
            center = (x + w // 2, y + h // 2)
            avg_color = self.get_average_color(frame, (x, y, w, h))
            object_id = self.match_object(center, avg_color, w, h)

            if object_id is not None and random.random() < self.disappear_prob:
                continue
            
            if object_id is not None:
                self.objects[object_id]['history'].append(center)
                self.objects[object_id]['avg_color'] = avg_color
                self.objects[object_id]['bbox'] = (x, y, w, h)
                self.objects[object_id]['skipped_frames'] = 0
                updated_ids.append(object_id)
            else:
                new_color = self.generate_unique_color()
                self.objects[self.next_id] = {
                    'history': [center],
                    'color': new_color,
                    'avg_color': avg_color,
                    'bbox': (x, y, w, h),
                    'skipped_frames': 0,
                    'last_direction': None
                }
                updated_ids.append(self.next_id)
                self.next_id += 1

        for obj_id in self.objects:
            if obj_id not in updated_ids:
                self.objects[obj_id]['skipped_frames'] += 1

        self.objects = {obj_id: data for obj_id, data in self.objects.items() if data['skipped_frames'] <= self.max_frames_to_skip}

    def match_object(self, center, avg_color, w, h):
        best_match, min_dist = None, self.max_distance
        for obj_id, obj_data in self.objects.items():
            last_center = obj_data['history'][-1]
            direction = np.array(center) - np.array(last_center)
            direction_norm = np.linalg.norm(direction)
            if direction_norm != 0:
                direction = direction / direction_norm
            
            if obj_data['last_direction'] is not None:
                direction_change = np.dot(direction, obj_data['last_direction'])
            else:
                direction_change = 1

            dist = np.linalg.norm(np.array(center) - np.array(last_center))
            color_diff = np.linalg.norm(np.array(avg_color) - np.array(obj_data['avg_color']))
            size_diff = np.linalg.norm(np.array([w, h]) - np.array([obj_data['bbox'][2], obj_data['bbox'][3]]))
            combined_diff = dist + 0.1 * color_diff + 0.1 * size_diff - 0.5 * direction_change

            if combined_diff < min_dist:
                best_match, min_dist = obj_id, combined_diff
                obj_data['last_direction'] = direction

        return best_match

    def generate_unique_color(self):
        while True:
            new_color = tuple(random.randint(0, 255) for _ in range(3))
            if new_color not in self.used_colors:
                self.used_colors.add(new_color)
                return new_color

    def draw_paths(self, frame):
        frame_height, frame_width = frame.shape[:2]

        for obj_id, obj_data in sorted(self.objects.items(), key=lambda item: item[0]):
            x, y, w, h = obj_data['bbox']

            if w > self.min_size and h > self.min_size:
                for i in range(1, len(obj_data['history'])):
                    if np.linalg.norm(np.array(obj_data['history'][i]) - np.array(obj_data['history'][i - 1])) < self.max_distance:
                        cv2.line(frame, obj_data['history'][i - 1], obj_data['history'][i], obj_data['color'], 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), obj_data['color'], 2)

                id_position = (x, y - 10) if y > 20 else (x, y + h + 20)
                coord_position = (x, y + h + 15) if y + h + 30 < frame_height else (x, y - 10)

                if x + w > frame_width - 60:
                    id_position = (x - 60, y - 10) if y > 20 else (x - 60, y + h + 20)
                    coord_position = (x - 60, y + h + 15) if y + h + 30 < frame_height else (x - 60, y - 10)

                if y + h > frame_height - 20:
                    id_position = (x, y - 20)
                    coord_position = (x, y - 35)

                if y < 20:
                    id_position = (x, y + h + 20)
                    coord_position = (x, y + h + 35)

                if coord_position[1] > frame_height - 5:
                    coord_position = (x, y - 15)

                if x < 20:
                    id_position = (x + w + 5, y - 10 if y > 20 else y + h + 20)
                    coord_position = (x + w + 5, y + h + 15 if y + h + 30 < frame_height else y - 10)

                cv2.putText(frame, f"ID: {obj_id}", id_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_data['color'], 1)
                cv2.putText(frame, f"({x}, {y})", coord_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_data['color'], 1)


    @staticmethod
    def get_average_color(frame, bbox):
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        avg_color = np.mean(roi, axis=(0, 1))
        return avg_color


def detect_objects(frame, bg_subtractor, min_contour_area=1000):
    fg_mask = bg_subtractor.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > min_contour_area]
    
    return objects

def process_video(video_path, warmup_frames=10, output_path=None):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=70, detectShadows=True)
    tracker = ObjectTracker()
    
    frame_count = 0

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count <= warmup_frames:
            bg_subtractor.apply(frame)
            continue
        
        objects = detect_objects(frame, bg_subtractor)
        tracker.update(objects, frame)

        print(f"Frame {frame_count}:")
        for obj_id, obj_data in tracker.objects.items():
            x, y, w, h = obj_data['bbox']
            print(f"  Object {obj_id}: (x={x}, y={y}, w={w}, h={h})")

        tracker.draw_paths(frame)

        if output_path:
            out.write(frame)
        
        cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="object tracker")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output_path", type=str, help="Path to save the output video", default=None)
    parser.add_argument("--warmup_frames", type=int, help="Number of warmup frames for background subtraction", default=10)
    args = parser.parse_args()

    process_video(args.video_path, args.warmup_frames, args.output_path)
