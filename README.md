
# Documentation for Object Tracker

## Description

This script is used to track objects in a video. It uses the OpenCV library to detect objects based on background subtraction, and then tracks their movement through the frames. It takes into account the possibility that objects may disappear for a short time and reappear while maintaining their identity.

## Dependencies

To run the script, you need to install the following libraries:

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- setuptools

You can install the required libraries by running this command:

```bash
 pip install -r requirements.txt
```

## How to Use

### 1. Preparation

The script requires you to specify the path to the video file. Make sure the video file is available and you have the correct path.

### 2. Running the Script

You can run the script from the command line. Example usage:

```bash
python tracker.py path/to/your/video.mp4 --output_path path/to/output.avi --warmup_frames 20
```

### 3. Parameters

The script supports the following parameters:

- `video_path`: The path to the video file that will be processed (required).
- `--output_path`: The path to save the output video file (optional). If not specified, the output video will not be saved.
- `--warmup_frames`: The number of warmup frames for background subtraction before starting to track objects (optional, default is `10`).

### 4. Example Command

```bash
python tracker.py luxonis_task_video.mp4 --output_path output_video.avi --warmup_frames 15
```

### 5. Code Explanation

- **Class `MultiObjectTracker`:** This class is responsible for detecting and tracking objects in video frames. Key methods include:
  - `update(detections, frame)`: Updates the tracker with detected objects
  - `match_object(center, avg_color, w, h)`: Matches new objects with those already being tracked
  - `generate_unique_color()`: Generates a unique color for each object
  - `draw_paths(frame)`: Draws the paths of tracked objects on the frame
  
- **Function `detect_objects(frame, bg_subtractor, min_contour_area=1000)`:** Detects objects in a frame based on background subtraction

- **Function `process_video(video_path, warmup_frames=10, output_path=None)`:** The main video processing loop that initializes the tracker, processes frames, and displays results. If `output_path` is specified, the result is saved as an output video

### 6. Visualization and Data Output

- The paths of objects will be visually shown on the video frames as colored lines.
- Each object’s ID and coordinates (x, y) are displayed on the video frames next to the bounding box of the object.
- The coordinates of detected objects will be printed to the console for each frame in the following format:
  ```
  Frame 1:
    Object 0: (x=100, y=200, w=50, h=50)
    Object 1: (x=300, y=400, w=60, h=60)
  ```

### 7. Stopping the Script

To stop the script and exit the video display, press `q` in the video window.

### 8. Improvements
- Consider advanced methods like deep learning detectors
- Add logic to instantly remove the bounding box when the object is no longer detected
- Reduce max_frames_to_skip or use Kalman filter predictions to handle this
- Use dynamic text placement to prevent coordinates and ID from overlapping
- Implement checks to avoid text collisions with the edges of the frame and other objects
- Introduce multithreading to process object detection and tracking in parallel, improving performance

