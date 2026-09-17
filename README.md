# D-PoSE ROS2 Webcam Demo - Usage Instructions



### Common Usage Examples

```bash
# Use external USB camera (camera ID 1)
python3 ros_demo_webcam.py --camera-id 1

# Enable live video display window
python3 ros_demo_webcam.py --display

# Use different camera resolution
python3 ros_demo_webcam.py --width 1920 --height 1080

# ArUco marker detection is on by default and is required for skeletons to
# publish; disable it to publish in the raw camera frame
python3 ros_demo_webcam.py --no-use-aruco

# Restrict to a specific marker ID/size for camera calibration
python3 ros_demo_webcam.py --aruco-marker-id 14 --aruco-marker-length 0.12

# Use custom model configuration
python3 ros_demo_webcam.py --cfg my_config.yaml --ckpt my_model.ckpt

# Lower detection threshold for more sensitive detection
python3 ros_demo_webcam.py --detection-threshold 0.5

# Combine multiple options
python3 ros_demo_webcam.py --camera-id 1 --display --fps 30
```

## Command Line Options

### Essential Options
- `--camera-id`: Camera device ID (default: 0)
- `--display`: Show live video window (use 'q' to quit)
- `--cfg`: Path to model configuration file
- `--ckpt`: Path to model checkpoint file

### Camera Configuration
- `--width`: Camera width in pixels (default: 1280)
- `--height`: Camera height in pixels (default: 720)  
- `--fps`: Frame rate (default: 13)

### Processing Options
- `--detection-threshold`: Person detection confidence (default: 0.7)
- `--detector`: Detector type - 'yolo' or 'maskrcnn' (default: maskrcnn)

### Additional Features
- `--use-aruco` / `--no-use-aruco`: Enable/disable ArUco marker detection (default: enabled). While
  enabled, skeletons are published relative to the marker and only after it has been seen at least once
- `--aruco-marker-id`: ArUco marker ID (DICT_6X6_250) to track (default: any marker)
- `--aruco-marker-length`: Printed marker side length in meters (default: 0.15)
- `--output-folder`: Directory for log files (default: ./logs)

### Static Camera Averaging
With a fixed camera the marker pose only changes when the robot moves along its slider, so
observations at the same slider position can be averaged into one low-noise pose.

- `--static-camera`: Average marker observations per slider position. Detection stops once a
  position has converged (re-checked every few seconds), the averaged pose keeps publishing while
  the marker is occluded, and returning to a known position reuses what was learned there
- `--static-robot`: The robot never moves, so everything is averaged into a single position and no
  slider topic is needed
- `--slider-topic`: `std_msgs/Float64` topic with the slider position in meters (default: `/slider/position_y`)
- `--slider-bin`: Slider positions within this distance share one averaged pose (default: 0.01 m)
- `--marker-table-file`: Where learned poses are saved on shutdown and reloaded on start
  (default: `<output-folder>/aruco_marker_table.json`)
- `--marker-min-samples`: Observations needed before a position is trusted (default: 30)
- `--marker-spread-threshold`: Required accuracy of the averaged position (default: 0.01 m)
- `--marker-recheck-interval`: How often a converged position is re-detected (default: 5 s)
- `--marker-alarm-pct`: After loading from disk, warn while live observations disagree by more than
  this percentage of the marker distance (default: 5%). If most of them disagree, the stored pose is
  discarded and relearned

Learned positions are also fitted with a straight line (`marker pose = origin + slider x direction`),
since travel is 1D and the marker rides the carriage rigidly. A slider position that was never visited
is then interpolated from that line instead of being learned from scratch, and every observation
improves every position. Measured positions always win over the line, and the fit is refused if its
residual exceeds `--marker-fit-residual`, which covers a rail that is not straight in the camera's
view or a slider value that is not in meters.

- `--no-marker-line-fit`: Never interpolate, learn each slider position on its own
- `--marker-fit-residual`: Largest residual the fit may have before interpolation is refused
  (default: 0.02 m)

Without a slider publisher the node warns and falls back to a single moving average, since it cannot
tell whether the robot moved.

```bash
# Fixed camera, robot moving on its conveyor
python3 ros_demo_webcam.py --static-camera

# Fixed camera and robot parked
python3 ros_demo_webcam.py --static-camera --static-robot
```

## Getting Help

```bash
# View all available options with descriptions
python3 ros_demo_webcam.py --help
```

## ROS2 Topics Published

The script publishes to the following ROS2 topics:

- `/humans` (skeleton_msgs/Skeletons): 3D skeleton data for all detected persons
- **TF Transforms**: Joint positions as transforms for visualization in RViz

## Troubleshooting

### Common Issues

**Camera not found:**
```
Error: Cannot open camera 0
```
Solution: Try different camera IDs (--camera-id 1, 2, etc.) or check camera connections.

**CUDA not available:**
```
Warning: CUDA not available, using CPU (will be slower)
```
Solution: Install CUDA drivers and PyTorch with CUDA support, or accept slower CPU processing.

**Model files not found:**
```
Required file not found: data/ckpt/paper_arxiv.ckpt
```
Solution: Download the model checkpoint as described in the main README.md

**ROS2 not configured:**
```
ModuleNotFoundError: No module named 'rclpy'
```
Solution: Source your ROS2 setup: `source /opt/ros/humble/setup.bash`

### Performance Tips

1. **Use CUDA**: Ensure PyTorch is installed with CUDA support
2. **Adjust resolution**: Lower camera resolution for better FPS
3. **Adjust detection threshold**: Higher threshold = fewer false positives but may miss some people
4. **Close video display**: Disable `--display` for better performance in production

## Safety Notes

- Press 'q' in the video window to quit safely
- Use Ctrl+C in terminal as backup to stop the program
- Camera LED will turn off when program exits properly

## Integration with RViz

To visualize the pose estimation results:

1. Launch RViz: `rviz2`
2. Add TF display to see joint transforms
3. Set fixed frame to "Camera" or "base_link"
4. The script publishes transforms for all detected human joints
