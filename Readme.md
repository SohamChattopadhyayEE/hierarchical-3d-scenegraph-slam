# hierarchical-3d-scenegraph-slam

RGB-D dataset streaming, unsupervised instance segmentation (FastSAM /
MobileSAM), and 3D back-projection + EKF object tracking, built against ROS 2
Jazzy and the TUM `rgbd_dataset_freiburg1_desk` sequence.

## Setup

Run inside the ROS 2 Jazzy docker container (`./run_docker.sh` from the
workspace root).

```
pip install ultralytics
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
wget -O mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```
FastSAM weights (`FastSAM-s.pt`) auto-download on first run. All parameters
(topics, model paths, camera intrinsics, thresholds) live in
`config/params.yaml` -- edit it before running.

## 1. Stream the dataset over ROS topics

```
python3 utils/ros2_streamer.py --ros-args --params-file config/params.yaml
```
Publishes RGB, depth, odometry, IMU, and tf from the TUM dataset, replayed
at `playback_rate`.

## 2. Segment + track objects, live over ROS

In a second terminal, alongside the streamer:
```
python3 mapping_node.py
```
`config/params.yaml` is loaded automatically as the default parameter
source (no need to pass `--ros-args` yourself unless overriding something,
e.g. `python3 mapping_node.py --ros-args -p segmentor_type:=mobilesam`).

## 2b. Or run offline, no ROS

Same pipeline, reading directly from a TUM dataset folder instead of ROS
topics:
```
python3 mapping_from_file.py datasets/rgbd_dataset_freiburg1_desk
```

## Outputs

- `datasets/segmented_masks/<segmentor_type>/mask/*.png` + `mask.txt` --
  colorized instance masks per frame.
- `datasets/objects/<object_id>/trajectory.txt`, `*_points.npy`, `*.png` --
  per-object 3D trajectory, back-projected point cloud, and image crop.

## Testing / debugging

Needs `matplotlib` (all viewers) and `scipy` (the two reconstruction viewers):
```
pip install matplotlib scipy
```

```
# Run a segmentor directly on one dataset frame, no ROS -- saves the raw
# mask + a colorized visualization locally in test_modules/.
python3 test_modules/test_fastsam.py
python3 test_modules/test_mobilesam.py

# Single frame: one object's 3D points from one timestamp.
python3 test_modules/view_pointcloud.py datasets/objects/<id>/<ts>_points.npy

# Step through one object's frames with Right/Left: point cloud (textured
# from its crop) and the original image, side by side, updating together.
python3 test_modules/view_multiview_pointcloud.py datasets/objects/<id>

# Combine several objects' RECONSTRUCTED (denoised/densified, see below),
# textured point clouds into one shared-frame 3D scene.
python3 test_modules/view_objects_map.py datasets/objects <id1> <id2> ...

# One object: raw-fused vs. reconstruction_3d/reconstruct.py's cleaned
# reconstruction, both textured, side by side.
python3 test_modules/view_reconstructed_object.py datasets/objects/<id>
```
