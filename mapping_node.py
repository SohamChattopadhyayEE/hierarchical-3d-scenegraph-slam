#!/usr/bin/env python3
"""ROS 2 (Jazzy) mapping node.

Subscribes to the RGB, depth, odometry, and accelerometer topics published
by utils/ros2_streamer.py. For now, only RGB frames are routed through an
object segmentor (FastSAM or MobileSAM, selected by the segmentor_type
parameter) on a dedicated worker thread; the resulting per-object instance
masks are published and saved to disk. Depth, odometry and IMU are
subscribed and stored for future fusion but not yet processed.
"""

import os
import sys
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu

from object_segmentor.fastsam_segmentor import FastSAMSegmentor
from object_segmentor.mobilesam_segmentor import MobileSAMSegmentor
from object_segmentor.object_segmentor_base import ObjectSegmentorBase
from object_segmentor.visualization import colorize_depth, colorize_label_mask
from reconstruction_3d.back_projector import BackProjector
from reconstruction_3d.object_tracker import Tracker

# Loaded automatically by main() if the caller doesn't supply its own
# --ros-args --params-file / -p overrides (which still take precedence).
DEFAULT_PARAMS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'config', 'params.yaml')


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class MappingNode(Node):

    def __init__(self):
        # No declare_parameter calls: every parameter this node reads must
        # come from a params file / -p override (see DEFAULT_PARAMS_FILE),
        # auto-declared here from whatever override values were supplied.
        super().__init__('mapping_node', automatically_declare_parameters_from_overrides=True)

        segmentor_type = self.get_parameter('segmentor_type').value
        device = self.get_parameter('device').value

        if segmentor_type == 'fastsam':
            self.segmentor: ObjectSegmentorBase = FastSAMSegmentor(
                model_path=self.get_parameter('fastsam_model_path').value,
                conf_threshold=self.get_parameter('conf_threshold').value,
                iou_threshold=self.get_parameter('iou_threshold').value,
                device=device,
            )
        elif segmentor_type == 'mobilesam':
            self.segmentor = MobileSAMSegmentor(
                checkpoint_path=self.get_parameter('mobilesam_checkpoint_path').value,
                model_type=self.get_parameter('mobilesam_model_type').value,
                points_per_side=self.get_parameter('points_per_side').value,
                pred_iou_thresh=self.get_parameter('pred_iou_thresh').value,
                stability_score_thresh=self.get_parameter('stability_score_thresh').value,
                device=device,
            )
        else:
            raise ValueError(
                f"Unknown segmentor_type '{segmentor_type}' (expected 'fastsam' or 'mobilesam')")

        self.output_dir = os.path.join(
            self.get_parameter('output_dataset_dir').value, segmentor_type)
        os.makedirs(os.path.join(self.output_dir, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'depth_color'), exist_ok=True)
        self.index_file = open(os.path.join(self.output_dir, 'mask.txt'), 'w')
        self.index_file.write('# segmentation masks\n# timestamp filename\n')

        self.bridge = CvBridge()

        self.back_projector = BackProjector(
            fx=self.get_parameter('fx').value,
            fy=self.get_parameter('fy').value,
            cx=self.get_parameter('cx').value,
            cy=self.get_parameter('cy').value,
            erosion_kernel=self.get_parameter('mask_erosion_kernel').value,
            depth_discontinuity_thresh=self.get_parameter('depth_discontinuity_thresh').value,
        )
        self.tracker = Tracker(
            output_dir=self.get_parameter('objects_output_dir').value,
            gate=self.get_parameter('track_gate_m').value,
        )

        # _latest_odom is consumed for 3D back-projection/tracking below;
        # _latest_imu is reserved for future fusion, not yet consumed.
        self._latest_depth = None
        self._latest_odom = None
        self._latest_imu = None

        # Single-slot "latest frame wins" handoff to the worker thread: an
        # unbounded queue would only build backlog since inference is much
        # slower than the camera rate, so each wake picks up only the
        # newest frame and silently drops whatever arrived while busy.
        self._frame_lock = threading.Lock()
        self._frame_cv = threading.Condition(self._frame_lock)
        self._latest_frame = None  # (header, bgr_image)
        self._running = True

        self.mask_pub = self.create_publisher(
            Image, self.get_parameter('mask_topic').value, 10)

        self.create_subscription(
            Image, self.get_parameter('rgb_topic').value, self._on_rgb, 10)
        self.create_subscription(
            Image, self.get_parameter('depth_topic').value, self._on_depth, 10)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self._on_odom, 10)
        self.create_subscription(
            Imu, self.get_parameter('imu_topic').value, self._on_imu, 10)

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        self.get_logger().info(f"mapping_node started with segmentor_type='{segmentor_type}'")

    def _on_rgb(self, msg: Image):
        bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self._frame_cv:
            self._latest_frame = (msg.header, bgr_image)
            self._frame_cv.notify()

    def _on_depth(self, msg: Image):
        self._latest_depth = msg

    def _on_odom(self, msg: Odometry):
        self._latest_odom = msg

    def _on_imu(self, msg: Imu):
        self._latest_imu = msg

    def _worker_loop(self):
        while self._running:
            with self._frame_cv:
                self._frame_cv.wait_for(
                    lambda: self._latest_frame is not None or not self._running)
                if not self._running:
                    return
                header, bgr_image = self._latest_frame
                self._latest_frame = None

            label_mask = self.segmentor.segment(bgr_image)
            self._publish_mask(header, label_mask)
            # self._save_mask(header, (label_mask.astype(np.float32) / label_mask.max() * 255).astype(np.uint8))
            color = colorize_label_mask(label_mask)
            self._save_mask(header, color)

            depth_msg, odom_msg = self._latest_depth, self._latest_odom
            if depth_msg is not None and odom_msg is not None:
                depth_m = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
                self._save_depth_color(header, colorize_depth(depth_m))
                pose = odom_msg.pose.pose
                R = _quat_to_matrix(pose.orientation)
                t = np.array([pose.position.x, pose.position.y, pose.position.z])
                timestamp = header.stamp.sec + header.stamp.nanosec * 1e-9

                detections = self.back_projector.project(label_mask, depth_m)
                self.tracker.update(detections, R, t, timestamp, bgr_image, label_mask)

    def _publish_mask(self, header, label_mask: np.ndarray):
        msg = self.bridge.cv2_to_imgmsg(label_mask.astype(np.uint16), encoding='mono16')
        msg.header = header
        self.mask_pub.publish(msg)

    def _save_mask(self, header, label_mask: np.ndarray):
        stamp = header.stamp.sec + header.stamp.nanosec * 1e-9
        stamp_str = f'{stamp:.6f}'
        filename = os.path.join('mask', f'{stamp_str}.png')
        cv2.imwrite(os.path.join(self.output_dir, filename), label_mask)
        self.index_file.write(f'{stamp_str} {filename}\n')
        self.index_file.flush()

    def _save_depth_color(self, header, depth_color: np.ndarray):
        stamp_str = f'{header.stamp.sec + header.stamp.nanosec * 1e-9:.6f}'
        cv2.imwrite(os.path.join(self.output_dir, 'depth_color', f'{stamp_str}.png'), depth_color)

    def destroy_node(self):
        self._running = False
        with self._frame_cv:
            self._frame_cv.notify_all()
        self._worker_thread.join(timeout=2.0)
        self.index_file.close()
        super().destroy_node()


def _with_default_params_file(args):
    """Injects --ros-args --params-file DEFAULT_PARAMS_FILE as the base
    parameter source, ahead of whatever ROS args the caller supplied --
    any -p / --params-file the caller passes still overrides it, since
    later entries in the same --ros-args block win (rclpy/rcl's normal,
    documented precedence rule).
    """
    raw_args = list(sys.argv[1:] if args is None else args)
    if not os.path.isfile(DEFAULT_PARAMS_FILE):
        return raw_args

    if '--ros-args' in raw_args:
        idx = raw_args.index('--ros-args')
        prefix, ros_args = raw_args[:idx], raw_args[idx + 1:]
    else:
        prefix, ros_args = raw_args, []

    return prefix + ['--ros-args', '--params-file', DEFAULT_PARAMS_FILE] + ros_args


def main(args=None):
    # rclpy.init() expects a full argv-style list (argv[0] is the program
    # name and is skipped by rcl's argument parser), so a placeholder is
    # prepended regardless of how args was supplied.
    rclpy.init(args=['mapping_node'] + _with_default_params_file(args))
    node = MappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
