#!/usr/bin/env python3
"""Walks the pedestrian back and forth across the zebra crossing.

The model is static, so nothing in the physics engine moves it: this node
teleports it with /set_entity_state, the same fast path the traffic light node
uses to switch its lamps.

Cycle, by default:

    wait at the south kerb  ->  cross  ->  wait at the north kerb  ->  cross back

Crossing 900 mm at 0.14 m/s takes about 6.4 s, so with the default 15 s dwell a
full cycle is roughly 43 s and the road is clear for about 30 of them. That is
worth knowing when you set the grading runs: a robot that never handles the
crossing still gets through unchallenged most of the time.

Set trigger_range above zero to close that gap. The pedestrian then also steps
off the kerb when the robot comes within that distance, so every run is tested
once rather than by luck. It is off by default.

Parameters
----------
model            name of the model in the world           (pedestrian)
x                where it crosses, in world metres        (0.96)
y_near, y_far    the two kerbs                            (-1.95, -1.05)
speed            walking speed in m/s                     (0.14)
dwell            seconds waiting at each kerb             (15.0)
dwell_jitter     +/- seconds added at random to dwell     (0.0)
trigger_range    step out when the robot is this close    (0.0 = never)
seed             random seed                              (0)
"""

import math
import random
import signal

import rclpy
from gazebo_msgs.srv import SetEntityState
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

WAIT_NEAR, CROSS_OUT, WAIT_FAR, CROSS_BACK = range(4)


class Pedestrian(Node):

    def __init__(self):
        super().__init__('pedestrian')

        self.declare_parameter('model', 'pedestrian')
        self.declare_parameter('x', 0.96)
        self.declare_parameter('y_near', -1.95)
        self.declare_parameter('y_far', -1.05)
        self.declare_parameter('speed', 0.14)
        self.declare_parameter('dwell', 15.0)
        self.declare_parameter('dwell_jitter', 0.0)
        self.declare_parameter('trigger_range', 0.0)
        self.declare_parameter('seed', 0)

        g = self.get_parameter
        self.model = g('model').value
        self.x = float(g('x').value)
        self.y_near = float(g('y_near').value)
        self.y_far = float(g('y_far').value)
        self.speed = float(g('speed').value)
        self.dwell = float(g('dwell').value)
        self.jitter = float(g('dwell_jitter').value)
        self.trigger = float(g('trigger_range').value)
        self.rng = random.Random(int(g('seed').value))

        self.span = abs(self.y_far - self.y_near)
        self.cross_time = self.span / max(self.speed, 1e-3)

        self.state = WAIT_NEAR
        self.t0 = self.now()
        self.wait_for = self.draw_dwell()
        self.robot = None

        self.cli = self.create_client(SetEntityState, '/set_entity_state')
        self.create_subscription(Odometry, '/odom', self._on_odom, 1)

        if not self.cli.wait_for_service(timeout_sec=30.0):
            raise SystemExit('/set_entity_state never came up')

        self.place(self.y_near)
        self.create_timer(0.05, self.tick)

        self.get_logger().info(
            f'crossing at x={self.x:.2f} between y={self.y_near:.2f} and '
            f'y={self.y_far:.2f}, {self.cross_time:.1f} s per crossing, '
            f'{self.dwell:.0f} s at each kerb'
            + (f', steps out at {self.trigger:.2f} m' if self.trigger > 0 else ''))

    # ------------------------------------------------------------------ utils
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def draw_dwell(self):
        if self.jitter <= 0:
            return self.dwell
        return max(1.0, self.rng.uniform(self.dwell - self.jitter,
                                         self.dwell + self.jitter))

    def _on_odom(self, msg):
        # Only ever used to decide when to step out. /odom is the robot's own
        # drifting estimate, but for "is it nearby" that is good enough.
        p = msg.pose.pose.position
        self.robot = (p.x, p.y)

    def robot_is_close(self):
        if self.trigger <= 0 or self.robot is None:
            return False
        dx = self.robot[0] - self.x
        dy = self.robot[1] - (self.y_near + self.y_far) / 2.0
        return math.hypot(dx, dy) <= self.trigger

    def place(self, y):
        req = SetEntityState.Request()
        st = req.state
        st.name = self.model
        st.pose.position.x = self.x
        st.pose.position.y = float(y)
        st.pose.position.z = 0.0
        st.pose.orientation.w = 1.0
        st.reference_frame = 'world'
        # Fire and forget. Waiting on the future inside a timer callback would
        # deadlock the single-threaded executor.
        self.cli.call_async(req)

    # ------------------------------------------------------------------- loop
    def tick(self):
        t = self.now() - self.t0

        if self.state == WAIT_NEAR:
            if t >= self.wait_for or self.robot_is_close():
                self.step(CROSS_OUT)
        elif self.state == CROSS_OUT:
            if t >= self.cross_time:
                self.place(self.y_far)
                self.wait_for = self.draw_dwell()
                self.step(WAIT_FAR)
            else:
                self.place(self.lerp(self.y_near, self.y_far, t / self.cross_time))
        elif self.state == WAIT_FAR:
            if t >= self.wait_for:
                self.step(CROSS_BACK)
        elif self.state == CROSS_BACK:
            if t >= self.cross_time:
                self.place(self.y_near)
                self.wait_for = self.draw_dwell()
                self.step(WAIT_NEAR)
            else:
                self.place(self.lerp(self.y_far, self.y_near, t / self.cross_time))

    def step(self, new_state):
        self.state = new_state
        self.t0 = self.now()

    @staticmethod
    def lerp(a, b, f):
        return a + (b - a) * max(0.0, min(1.0, f))


# The shutdown dance below is the same one the other nodes use: install the
# SIGTERM handler BEFORE the node exists, then only set a flag. Raising from the
# handler lands inside a pybind11 call and turns into an unrelated error.
def catch_sigterm():
    stopping = {'now': False}
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(now=True))
    return stopping


def spin(node, stopping):
    while rclpy.ok() and not stopping['now']:
        try:
            rclpy.spin_once(node, timeout_sec=0.1)
        except Exception:
            if stopping['now'] or not rclpy.ok():
                break
            raise


def main(args=None):
    rclpy.init(args=args)
    stopping = catch_sigterm()
    node = Pedestrian()
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
