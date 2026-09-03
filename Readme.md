# hierarchical-3d-scenegraph-slam

RGB-D dataset streaming, unsupervised instance segmentation (FastSAM /
MobileSAM), 3D back-projection + EKF object tracking, and dense
reconstruction (Poisson / Gaussian Splatting / RANSAC planes), built against
ROS 2 Jazzy and the TUM `rgbd_dataset_freiburg1_desk` sequence.

## Setup

Run inside the ROS 2 Jazzy docker container (`./run_docker.sh` from the
workspace root).

```
pip install ultralytics                                              # FastSAM
pip install git+https://github.com/ChaoningZhang/MobileSAM.git       # MobileSAM
wget -O mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
pip install matplotlib scipy open3d                                  # viewers / reconstruction
```
FastSAM weights (`FastSAM-s.pt`) auto-download on first run. All parameters
(topics, model paths, camera intrinsics, thresholds, object-selection gates)
live in `config/params.yaml` -- edit it before running.

Gaussian Splatting is real, trained 3DGS and additionally needs a CUDA GPU,
`gsplat`, and the CUDA toolkit compiler (`torch` ships a CUDA *runtime* but
not `nvcc`, which `gsplat` needs to build its kernels):
```
pip install gsplat
apt-get -y install cuda-nvcc-12-6 cuda-cudart-dev-12-6 cuda-crt-12-6
export PATH=/usr/local/cuda-12.6/bin:$PATH
```

## Pipeline

### 1. Stream the dataset over ROS topics
Publishes RGB, depth, odometry, IMU and tf from the TUM dataset, replayed at
`playback_rate`.
```
python3 utils/ros2_streamer.py --ros-args --params-file config/params.yaml
```

### 2. Segment + track objects, live over ROS
Segments each frame, back-projects masked depth into 3D, tracks object
centroids in world frame with an EKF, writes per-object output. In a second
terminal, alongside the streamer:
```
python3 mapping_node.py
```
`config/params.yaml` loads automatically (no `--ros-args` needed unless
overriding, e.g. `--ros-args -p segmentor_type:=mobilesam`).

### 2b. Or run the same pipeline offline, no ROS
Reads a TUM dataset folder directly instead of subscribing to topics.
```
python3 mapping_from_file.py datasets/rgbd_dataset_freiburg1_desk
```

### Outputs
- `<output_dataset_dir>/<segmentor_type>/mask/*.png` + `mask.txt` --
  colorized instance masks per frame.
- `<output_dataset_dir>/<segmentor_type>/depth_color/*.png` -- jet-colormapped
  depth, for eyeballing only.
- `<objects_output_dir>/<object_id>/` -- per tracked object:
  `trajectory.txt` (timestamped world centroid), `<ts>_points.npy`
  (world-frame point cloud), `<ts>.png` (image crop), `<ts>_pose.npz`
  (camera pose + crop bbox, needed for Gaussian Splatting).

Only "significant" objects are written: masks under `min_area` px are never
back-projected, and tracks shorter than `min_track_length` frames are never
saved (both in `config/params.yaml`).

## Dense reconstruction / map building

### Combined multi-object map (one joint Gaussian Splat fit)
Pools several objects' points and posed views, cleans them as one scene, and
fits ONE Gaussian Splat model across all of them -- smoother and less seamy
than reconstructing each object separately. Saves `combined_map.ply`.
```
python3 test_modules/combined_complete_map.py datasets/objects 22 25 27 30
python3 test_modules/combined_complete_map.py datasets/objects --all
```
`--all` uses every object folder found. Options: `--iters`, `--device`,
`--output`.

### Whole-scene map straight from RGB-D (no segmentation/tracking)
Same technique, but sourced directly from `rgb.txt`/`depth.txt`/
`groundtruth.txt` using whole frames instead of tracked object crops. Saves
`gs_map.ply` next to the dataset.
```
python3 gs_map_from_rgbd.py datasets/rgbd_dataset_freiburg1_desk
python3 gs_map_from_rgbd.py datasets/rgbd_dataset_freiburg1_desk --stride 10 --iters 4000
```
`--stride` picks every Nth frame (default 5) -- whole-frame back-projection
across hundreds of frames adds up fast.

### Walls / floor via RANSAC plane extraction
Takes the BLACK (non-object) pixels of each saved mask, back-projects that
depth, and RANSAC-fits large planes. Only physically large surfaces are kept
(both in-plane dimensions must reach `--min_extent`), so desk tops and
clutter slabs are rejected. Saves `planes.ply`.
```
# extraction only, one color per plane (verify extraction by eye)
python3 reconstruction_3d/plane_reconstruction.py \
    datasets/rgbd_dataset_freiburg1_desk \
    datasets/segmented_masks/fastsam/mask

python3 ... --texture          # paint planes with their real image colors
python3 ... --min_extent 1.5   # stricter: only really big surfaces
python3 ... --gs               # also fit Gaussian Splatting to the planes
```
Gaussian Splatting is off by default here; extraction runs standalone.

## Viewers / debugging

### Segmentors, on a single frame (no ROS)
Runs one segmentor on one dataset image, saves the raw mask + a colorized
visualization into `test_modules/`.
```
python3 test_modules/test_fastsam.py
python3 test_modules/test_mobilesam.py
```

### One object's point cloud, one frame
```
python3 test_modules/view_pointcloud.py datasets/objects/<id>/<ts>_points.npy
```

### Step through one object frame by frame
Right/Left arrows step through frames; shows the point cloud (textured from
that frame's crop) and the original image side by side.
```
python3 test_modules/view_multiview_pointcloud.py datasets/objects/<id>
```

### One object: raw vs. cleaned vs. dense reconstruction
Raw-fused vs. cleaned point cloud side by side, then its Poisson mesh, then
its trained Gaussian Splat, each in its own window.
```
python3 test_modules/view_reconstructed_object.py datasets/objects/<id>
```

### Several objects together in one scene
Each object's cleaned point cloud in one shared-frame scene, then all their
Poisson meshes, then all their Gaussian splats.
```
python3 test_modules/view_objects_map.py datasets/objects <id1> <id2> ...
```

### Maps with 3D bounding boxes per object
Overlays an oriented bounding box per object (fit to its cleaned points) on
top of any number of maps -- `planes.ply`, `combined_map.ply`, `gs_map.ply`,
... -- shown together, since they all share the map frame.
```
python3 test_modules/view_map_with_boxes.py datasets/objects 3 7 12 \
    --ply datasets/objects/combined_map.ply

python3 test_modules/view_map_with_boxes.py datasets/objects --all \
    --ply datasets/objects/combined_map.ply planes.ply
```
`--all` boxes every object folder found. `--ply` is optional (omit it to see
just the boxes).

### View any .ply, or several overlaid together
Each file keeps its own colors; all of them share the same map frame, so
they line up.
```
python3 test_modules/view_ply.py datasets/objects/combined_map.ply
python3 test_modules/view_ply.py planes.ply combined_map.ply gs_map.ply
```
