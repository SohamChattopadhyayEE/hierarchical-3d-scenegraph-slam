#!/usr/bin/env python3
"""ROS 2 (Jazzy) node that streams a TUM-RGBD style dataset.

Reads rgb.txt, depth.txt, groundtruth.txt and accelerometer.txt from a
dataset directory (e.g. rgbd_dataset_freiburg1_desk) and republishes the
recorded data on ROS topics, preserving the original relative timing
between all four streams.
"""

import os
import threading
import time

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def stamp_from_seconds(ts: float) -> TimeMsg:
    sec = int(ts)
    nanosec = int(round((ts - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    msg = TimeMsg()
    msg.sec = sec
    msg.nanosec = nanosec
    return msg


def read_tum_file(path: str):
    """Parse a whitespace-separated TUM-format text file, skipping comments."""
    entries = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            entries.append((float(fields[0]), fields[1:]))
    return entries


class DatasetStreamer(Node):

    def __init__(self):
        super().__init__('dataset_streamer')

        default_dataset = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../datasets/rgbd_dataset_freiburg1_desk',
        )

        self.declare_parameter('dataset_path', default_dataset)
        self.declare_parameter('playback_rate', 1.0)
        self.declare_parameter('loop', False)
        self.declare_parameter('depth_scale', 5000.0)
        self.declare_parameter('rgb_topic', 'camera/rgb/image_raw')
        self.declare_parameter('depth_topic', 'camera/depth/image_raw')
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('imu_topic', 'imu/data_raw')
        self.declare_parameter('camera_frame_id', 'camera_rgb_optical_frame')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('imu_frame_id', 'imu_link')
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('publish_tf', True)

        self.dataset_path = self.get_parameter('dataset_path').value
        self.playback_rate = float(self.get_parameter('playback_rate').value)
        self.loop = bool(self.get_parameter('loop').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.camera_frame_id = self.get_parameter('camera_frame_id').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.imu_frame_id = self.get_parameter('imu_frame_id').value
        self.map_frame_id = self.get_parameter('map_frame_id').value
        self.publish_tf = bool(self.get_parameter('publish_tf').value)

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.rgb_pub = self.create_publisher(
            Image, self.get_parameter('rgb_topic').value, 10)
        self.depth_pub = self.create_publisher(
            Image, self.get_parameter('depth_topic').value, 10)
        self.odom_pub = self.create_publisher(
            Odometry, self.get_parameter('odom_topic').value, 10)
        self.imu_pub = self.create_publisher(
            Imu, self.get_parameter('imu_topic').value, 10)

        if self.publish_tf:
            self._publish_static_transforms()

        self._stop_event = threading.Event()
        self._playback_thread = None

        self.events = self._load_events()
        if not self.events:
            self.get_logger().error(
                f"No data loaded from dataset path '{self.dataset_path}'")
            return

        self.get_logger().info(
            f"Loaded {len(self.events)} events from '{self.dataset_path}' "
            f"(playback_rate={self.playback_rate}, loop={self.loop})")

        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True)
        self._playback_thread.start()

    def _identity_transform(self, parent_frame, child_frame):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.rotation.w = 1.0
        return t

    def _publish_static_transforms(self):
        # The dataset does not provide camera/IMU extrinsics relative to the
        # body frame, so camera and IMU are treated as co-located with
        # base_link. Groundtruth is a global, drift-free trajectory, so
        # odom is treated as coincident with map.
        transforms = [
            self._identity_transform(self.map_frame_id, self.odom_frame_id),
            self._identity_transform(self.base_frame_id, self.camera_frame_id),
            self._identity_transform(self.base_frame_id, self.imu_frame_id),
        ]
        self.static_tf_broadcaster.sendTransform(transforms)

    def _load_events(self):
        required = ['rgb.txt', 'depth.txt', 'groundtruth.txt', 'accelerometer.txt']
        for name in required:
            full = os.path.join(self.dataset_path, name)
            if not os.path.isfile(full):
                self.get_logger().error(f"Missing required file: {full}")
                return []

        events = []
        for ts, fields in read_tum_file(os.path.join(self.dataset_path, 'rgb.txt')):
            events.append((ts, 'rgb', fields[0]))
        for ts, fields in read_tum_file(os.path.join(self.dataset_path, 'depth.txt')):
            events.append((ts, 'depth', fields[0]))
        for ts, fields in read_tum_file(os.path.join(self.dataset_path, 'groundtruth.txt')):
            events.append((ts, 'gt', [float(x) for x in fields]))
        for ts, fields in read_tum_file(os.path.join(self.dataset_path, 'accelerometer.txt')):
            events.append((ts, 'acc', [float(x) for x in fields]))

        events.sort(key=lambda e: e[0])
        return events

    def _playback_loop(self):
        while rclpy.ok() and not self._stop_event.is_set():
            t0_dataset = self.events[0][0]
            t0_wall = time.monotonic()

            for ts, etype, payload in self.events:
                if not rclpy.ok() or self._stop_event.is_set():
                    return
                target_wall = t0_wall + (ts - t0_dataset) / self.playback_rate
                delay = target_wall - time.monotonic()
                if delay > 0:
                    self._stop_event.wait(delay)
                self._publish_event(ts, etype, payload)

            if not self.loop:
                self.get_logger().info('Dataset playback finished.')
                return
            self.get_logger().info('Looping dataset playback.')

    def _publish_event(self, ts, etype, payload):
        stamp = stamp_from_seconds(ts)
        if etype == 'rgb':
            self._publish_image(stamp, payload, self.rgb_pub, depth=False)
        elif etype == 'depth':
            self._publish_image(stamp, payload, self.depth_pub, depth=True)
        elif etype == 'gt':
            self._publish_odom(stamp, payload)
        elif etype == 'acc':
            self._publish_imu(stamp, payload)

    def _publish_image(self, stamp, rel_path, publisher, depth):
        full_path = os.path.join(self.dataset_path, rel_path)
        if depth:
            raw = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                self.get_logger().warn(f"Could not read depth image: {full_path}")
                return
            meters = raw.astype(np.float32) / self.depth_scale
            msg = self.bridge.cv2_to_imgmsg(meters, encoding='32FC1')
        else:
            img = cv2.imread(full_path, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().warn(f"Could not read rgb image: {full_path}")
                return
            msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')

        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame_id
        publisher.publish(msg)

    def _publish_odom(self, stamp, fields):
        tx, ty, tz, qx, qy, qz, qw = fields
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame_id
        msg.child_frame_id = self.base_frame_id
        msg.pose.pose.position.x = tx
        msg.pose.pose.position.y = ty
        msg.pose.pose.position.z = tz
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        self.odom_pub.publish(msg)

        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame_id
            t.child_frame_id = self.base_frame_id
            t.transform.translation.x = tx
            t.transform.translation.y = ty
            t.transform.translation.z = tz
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(t)

    def _publish_imu(self, stamp, fields):
        ax, ay, az = fields
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self.imu_frame_id
        # No orientation or angular velocity in this dataset.
        msg.orientation_covariance[0] = -1.0
        msg.angular_velocity_covariance[0] = -1.0
        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az
        self.imu_pub.publish(msg)

    def destroy_node(self):
        self._stop_event.set()
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DatasetStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
