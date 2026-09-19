#!/usr/bin/env python3
"""
Save frames from the camera topics as PNG files.

Useful when running without the Gazebo window (headless), and for
collecting evidence to put in your report.

    ros2 run crc_sim snapshot
    ros2 run crc_sim snapshot --ros-args \
        -p out_dir:=/tmp/crc_shots -p period:=2.0 -p count:=5

Parameters:
    topics   : list of image topics  (default: sky camera + robot camera)
    out_dir  : output directory      (default /tmp/crc_shots)
    period   : seconds between shots (default 2.0)
    count    : how many shots before exiting; 0 = run forever (default 5)
"""

import os
import signal

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

try:
    import cv2
    from cv_bridge import CvBridge
except ImportError as e:                            
    raise SystemExit(
        'cv_bridge / opencv missing. Install with:\n'
        '  sudo apt install ros-humble-cv-bridge python3-opencv\n'
        f'({e})')


class Snapshot(Node):

    def __init__(self):
        super().__init__('snapshot')
        self.declare_parameter(
            'topics', ['/sky_cam/image_raw', '/camera/image_raw'])
        self.declare_parameter('out_dir', '/tmp/crc_shots')
        self.declare_parameter('period', 2.0)
        self.declare_parameter('count', 5)

        self.topics = list(self.get_parameter('topics').value)
        self.out_dir = self.get_parameter('out_dir').value
        self.count = int(self.get_parameter('count').value)
        period = float(self.get_parameter('period').value)

        os.makedirs(self.out_dir, exist_ok=True)
        self.bridge = CvBridge()
        self.latest = {}
        self.shot = 0

        for t in self.topics:
            self.create_subscription(
                Image, t,
                lambda msg, topic=t: self.latest.__setitem__(topic, msg),
                qos_profile_sensor_data)

        self.create_timer(period, self.grab)
        self.get_logger().info(
            f'Capturing {self.topics} -> {self.out_dir} every {period}s '
            f'({"unlimited" if self.count == 0 else self.count} shots)')

    def grab(self):
        missing = [t for t in self.topics if t not in self.latest]
        if missing:
            self.get_logger().warn(f'No data yet from: {missing}')
            return

        self.shot += 1
        for t in self.topics:
            name = t.strip('/').replace('/', '_')
            path = os.path.join(self.out_dir, f'{name}_{self.shot:02d}.png')
            img = self.bridge.imgmsg_to_cv2(self.latest[t], 'bgr8')
            cv2.imwrite(path, img)
            self.get_logger().info(f'  saved {path}  ({img.shape[1]}x{img.shape[0]})')

        if self.count and self.shot >= self.count:
            self.get_logger().info('Done.')
            raise SystemExit(0)


def catch_sigterm():
    """Turn SIGTERM into a flag instead of letting rclpy tear down the
    context. Must run before the node is built.
    """
    stopping = {'now': False}
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(now=True))
    return stopping


def spin(node, stopping):
    while rclpy.ok() and not stopping['now']:
        try:
            rclpy.spin_once(node, timeout_sec=0.1)
        except Exception:
            # Shutting down mid-callback is normal; anything else is not.
            if stopping['now'] or not rclpy.ok():
                break
            raise


def main(args=None):
    rclpy.init(args=args)
    stopping = catch_sigterm()
    node = Snapshot()
    try:
        spin(node, stopping)
    except (KeyboardInterrupt, SystemExit,
            ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
