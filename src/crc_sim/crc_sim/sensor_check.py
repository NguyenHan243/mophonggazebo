#!/usr/bin/env python3
"""
Quick check: are the simulated sensors publishing?

    ros2 run crc_sim sensor_check

Prints once per second: the rate and a short summary for /scan,
/camera/image_raw, /odom and /imu.
A line showing "NO DATA YET" means the matching Gazebo plugin is not running.
"""

import math
import signal
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, LaserScan


class SensorCheck(Node):

    def __init__(self):
        super().__init__('sensor_check')
        self.set_parameters(
            [Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        self.stat = {k: {'n': 0, 'last': None, 'info': '-'}
                     for k in ('scan', 'image', 'odom', 'imu')}

        self.create_subscription(LaserScan, '/scan', self.cb_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/image_raw', self.cb_img,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.cb_odom, 10)
        self.create_subscription(Imu, '/imu', self.cb_imu,
                                 qos_profile_sensor_data)

        self.t0 = time.time()
        self.create_timer(1.0, self.report)
        self.get_logger().info('Listening to sensors... (Ctrl+C to quit)')

    def _hit(self, key, info):
        s = self.stat[key]
        s['n'] += 1
        s['last'] = time.time()
        s['info'] = info

    def cb_scan(self, msg: LaserScan):
        valid = [r for r in msg.ranges if math.isfinite(r)]
        front = msg.ranges[0] if msg.ranges else float('nan')
        self._hit('scan', f'{len(msg.ranges)} rays, valid={len(valid)}, '
                          f'front={front:.2f} m')

    def cb_img(self, msg: Image):
        self._hit('image', f'{msg.width}x{msg.height} {msg.encoding}')

    def cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        v = msg.twist.twist
        self._hit('odom', f'x={p.x:+.2f} y={p.y:+.2f} '
                          f'v={v.linear.x:+.2f} w={v.angular.z:+.2f}')

    def cb_imu(self, msg: Imu):
        a = msg.linear_acceleration
        self._hit('imu', f'az={a.z:+.2f} m/s^2')

    def report(self):
        dt = max(1e-6, time.time() - self.t0)
        print('\n' + '=' * 74)
        print(f'{"TOPIC":<24}{"Hz":>7}  DETAILS')
        print('-' * 74)
        for key, topic in (('scan', '/scan'),
                           ('image', '/camera/image_raw'),
                           ('odom', '/odom'),
                           ('imu', '/imu')):
            s = self.stat[key]
            if s['n'] == 0:
                print(f'{topic:<24}{"--":>7}  !! NO DATA YET')
            else:
                print(f'{topic:<24}{s["n"] / dt:>7.1f}  {s["info"]}')
        print('=' * 74)


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
    node = SensorCheck()
    try:
        spin(node, stopping)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
